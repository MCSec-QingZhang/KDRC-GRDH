import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import os
import argparse
import csv
import math
import random
import zlib

# =========================================================
# Quad-2bit RDH core
# - fixed-length 4-pixel mapping
# - each 4-pixel group embeds 2 bits
# - extraction MUST use cover_extract_dir
#
# Gray mapping:
#   00 -> (+1,-1, 0, 0)
#   01 -> ( 0, 0,+1,-1)
#   11 -> (-1,+1, 0, 0)
#   10 -> ( 0, 0,-1,+1)
#
# To avoid overflow, cover is clipped to [1,254].
# =========================================================

COVER_CLIP_MIN = 1
COVER_CLIP_MAX = 254
CRC_BITS = 32


def clip_cover_rgb_u8(cover_rgb_u8: np.ndarray,
                      clip_min: int = COVER_CLIP_MIN,
                      clip_max: int = COVER_CLIP_MAX) -> np.ndarray:
    if cover_rgb_u8.dtype != np.uint8:
        raise ValueError("cover_rgb_u8 必须是 uint8。")
    if cover_rgb_u8.ndim != 3 or cover_rgb_u8.shape[2] != 3:
        raise ValueError("cover_rgb_u8 必须是 RGB 图像 (H,W,3)。")
    if not (0 <= clip_min <= clip_max <= 255):
        raise ValueError("clip range 非法。")
    return np.clip(cover_rgb_u8, np.uint8(clip_min), np.uint8(clip_max)).astype(np.uint8)


def clip_cover_ch_u8(cover_ch_u8: np.ndarray,
                     clip_min: int = COVER_CLIP_MIN,
                     clip_max: int = COVER_CLIP_MAX) -> np.ndarray:
    if cover_ch_u8.dtype != np.uint8:
        raise ValueError("cover_ch_u8 必须是 uint8。")
    if cover_ch_u8.ndim != 2:
        raise ValueError("cover_ch_u8 必须是二维通道图像。")
    if not (0 <= clip_min <= clip_max <= 255):
        raise ValueError("clip range 非法。")
    return np.clip(cover_ch_u8, np.uint8(clip_min), np.uint8(clip_max)).astype(np.uint8)


def pad_bits_to_multiple_of_6(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    4像素组嵌2bit；3通道均分时，总长度需要补到 6 的倍数。
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits 必须仅包含 0/1。")
    orig_len = int(bits.size)
    pad = (-orig_len) % 6
    if pad > 0:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)], axis=0)
    return bits, orig_len


def capacity_quad2bit_channel(ch_u8: np.ndarray) -> int:
    """
    每 4 个像素嵌入 2 bit。
    返回通道可嵌入的 bit 数。
    """
    if ch_u8.dtype != np.uint8 or ch_u8.ndim != 2:
        raise ValueError("ch_u8 必须是 uint8 的二维数组 (H,W)。")
    groups = int(ch_u8.size // 4)
    return 2 * groups


def bits2_to_symbol_index(b0: int, b1: int) -> int:
    """
    Gray mapping:
      00 -> 0
      01 -> 1
      11 -> 2
      10 -> 3
    """
    pair = (int(b0), int(b1))
    mapping = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    return mapping[pair]


def symbol_index_to_bits2(idx: int) -> Tuple[int, int]:
    mapping = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
    }
    return mapping[int(idx)]


TEMPLATES = np.array([
    [+1, -1,  0,  0],  # 00
    [ 0,  0, +1, -1],  # 01
    [-1, +1,  0,  0],  # 11
    [ 0,  0, -1, +1],  # 10
], dtype=np.int16)


def embed_quad2bit_channel(cover_ch_u8: np.ndarray,
                           payload_bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    单通道 4像素->2bit 固定长度嵌入。
    """
    if cover_ch_u8.dtype != np.uint8 or cover_ch_u8.ndim != 2:
        raise ValueError("cover_ch_u8 必须是 uint8 的二维数组 (H,W)。")

    cover_ch_u8 = clip_cover_ch_u8(cover_ch_u8)
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).flatten()

    if np.any((payload_bits != 0) & (payload_bits != 1)):
        raise ValueError("payload_bits 必须仅包含 0/1。")
    if payload_bits.size % 2 != 0:
        raise ValueError("quad2bit 单通道 payload_bits 长度必须为偶数。")

    cap_bits = capacity_quad2bit_channel(cover_ch_u8)
    if payload_bits.size > cap_bits:
        raise ValueError(f"payload_bits 长度={payload_bits.size} 超出通道容量={cap_bits}（每4像素2bit）。")

    flat = cover_ch_u8.reshape(-1).astype(np.int16)
    stego = flat.copy()

    num_groups = payload_bits.size // 2
    for g in range(num_groups):
        j = 4 * g
        idx = bits2_to_symbol_index(payload_bits[2 * g], payload_bits[2 * g + 1])
        stego[j:j+4] = flat[j:j+4] + TEMPLATES[idx]

    # 经过 clip 到 [1,254] 后，再加模板不会越界
    stego = stego.astype(np.uint8).reshape(cover_ch_u8.shape)
    return stego, int(payload_bits.size)


def extract_quad2bit_channel(stego_ch_u8: np.ndarray,
                             cover_ch_u8: np.ndarray,
                             payload_len: int) -> np.ndarray:
    """
    单通道提取：对每个 4 像素组，与 4 个模板做欧氏距离判决，恢复 2bit/group。
    提取时显式使用 cover_extract 对应通道。
    """
    if payload_len < 0:
        raise ValueError("payload_len 必须 >= 0。")
    if payload_len % 2 != 0:
        raise ValueError("quad2bit 单通道 payload_len 必须为偶数。")
    if stego_ch_u8.dtype != np.uint8 or cover_ch_u8.dtype != np.uint8:
        raise ValueError("stego_ch_u8 / cover_ch_u8 必须是 uint8。")
    if stego_ch_u8.shape != cover_ch_u8.shape or stego_ch_u8.ndim != 2:
        raise ValueError("stego_ch_u8 / cover_ch_u8 必须同尺寸二维数组。")

    cover_ch_u8 = clip_cover_ch_u8(cover_ch_u8)

    flat_s = stego_ch_u8.reshape(-1).astype(np.int16)
    flat_c = cover_ch_u8.reshape(-1).astype(np.int16)

    cap_bits = 2 * int(flat_c.size // 4)
    if payload_len > cap_bits:
        raise ValueError(f"payload_len={payload_len} 超出通道可提取容量={cap_bits}。")

    out = np.zeros((payload_len,), dtype=np.uint8)
    num_groups = payload_len // 2

    for g in range(num_groups):
        j = 4 * g
        d = np.array([
            flat_s[j + 0] - flat_c[j + 0],
            flat_s[j + 1] - flat_c[j + 1],
            flat_s[j + 2] - flat_c[j + 2],
            flat_s[j + 3] - flat_c[j + 3],
        ], dtype=np.int16)

        # 4 个模板中选最近者
        dist = np.sum((TEMPLATES - d) * (TEMPLATES - d), axis=1)
        idx = int(np.argmin(dist))
        b0, b1 = symbol_index_to_bits2(idx)
        out[2 * g] = b0
        out[2 * g + 1] = b1

    return out


def embed_quad2bit_rgb_even(cover_rgb_u8: np.ndarray,
                            payload_bits: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], int]:
    """
    三通道均分：
      - 先将 payload 补齐到 6 的倍数
      - 每通道得到相同长度 per_len_bits（偶数）
      - 每通道使用固定长度 quad2bit 映射
    """
    if cover_rgb_u8.dtype != np.uint8 or cover_rgb_u8.ndim != 3 or cover_rgb_u8.shape[2] != 3:
        raise ValueError("cover_rgb_u8 必须是 uint8 的 RGB 图像数组 (H,W,3)。")

    cover_rgb_u8 = clip_cover_rgb_u8(cover_rgb_u8)
    payload_pad, orig_len = pad_bits_to_multiple_of_6(payload_bits)
    per_len_bits = int(payload_pad.size // 3)

    if per_len_bits % 2 != 0:
        raise RuntimeError("内部错误：每通道 bit 数必须为偶数。")

    R, G, B = cover_rgb_u8[:, :, 0], cover_rgb_u8[:, :, 1], cover_rgb_u8[:, :, 2]
    capR, capG, capB = capacity_quad2bit_channel(R), capacity_quad2bit_channel(G), capacity_quad2bit_channel(B)

    if per_len_bits > capR or per_len_bits > capG or per_len_bits > capB:
        raise ValueError(
            "无法按三通道均匀分配嵌入：\n"
            f"  每通道需要 {per_len_bits} bits（payload 补齐后总长 {payload_pad.size}）\n"
            f"  通道容量：R={capR}, G={capG}, B={capB}\n"
            "请减少 payload 长度或更换图像。"
        )

    bits_R = payload_pad[:per_len_bits]
    bits_G = payload_pad[per_len_bits:2 * per_len_bits]
    bits_B = payload_pad[2 * per_len_bits:3 * per_len_bits]

    stego_R, embR = embed_quad2bit_channel(R, bits_R)
    stego_G, embG = embed_quad2bit_channel(G, bits_G)
    stego_B, embB = embed_quad2bit_channel(B, bits_B)

    if not (embR == per_len_bits and embG == per_len_bits and embB == per_len_bits):
        raise RuntimeError("某通道未嵌满，说明容量判定或实现不一致。")

    stego_rgb_u8 = np.stack([stego_R, stego_G, stego_B], axis=2).astype(np.uint8)
    return stego_rgb_u8, {"R": per_len_bits, "G": per_len_bits, "B": per_len_bits}, orig_len


def extract_quad2bit_rgb_even(stego_rgb_u8: np.ndarray,
                              cover_rgb_u8: np.ndarray,
                              embedded_per_channel: Dict[str, int],
                              orig_len: int) -> np.ndarray:
    if stego_rgb_u8.dtype != np.uint8 or cover_rgb_u8.dtype != np.uint8:
        raise ValueError("stego_rgb_u8 / cover_rgb_u8 必须是 uint8。")
    if stego_rgb_u8.shape != cover_rgb_u8.shape or stego_rgb_u8.ndim != 3 or stego_rgb_u8.shape[2] != 3:
        raise ValueError("stego / cover 必须同尺寸 RGB (H,W,3)。")

    cover_rgb_u8 = clip_cover_rgb_u8(cover_rgb_u8)
    per_len_bits = int(embedded_per_channel["R"])
    if not (embedded_per_channel.get("G") == per_len_bits and embedded_per_channel.get("B") == per_len_bits):
        raise ValueError("embedded_per_channel 必须满足 R/G/B 三者相同（均匀分配）。")

    bits_R = extract_quad2bit_channel(stego_rgb_u8[:, :, 0], cover_rgb_u8[:, :, 0], per_len_bits)
    bits_G = extract_quad2bit_channel(stego_rgb_u8[:, :, 1], cover_rgb_u8[:, :, 1], per_len_bits)
    bits_B = extract_quad2bit_channel(stego_rgb_u8[:, :, 2], cover_rgb_u8[:, :, 2], per_len_bits)

    bits_all = np.concatenate([bits_R, bits_G, bits_B], axis=0)
    return bits_all[:orig_len].astype(np.uint8)


# ============================================================
# BCH implementation (shortened binary BCH, narrow-sense BCH)
# ============================================================

PRIM_POLY = {
    2: 0x7,
    3: 0xB,
    4: 0x13,
    5: 0x25,
    6: 0x43,
    7: 0x83,
    8: 0x11D,
    9: 0x211,
    10: 0x409,
    11: 0x805,
    12: 0x1053,
    13: 0x201B,
    14: 0x402B,
    15: 0x8003,
}


def bits_to_int(bits: List[int]) -> int:
    x = 0
    for i, b in enumerate(bits):
        if b:
            x |= (1 << i)
    return x


def int_to_bits(x: int, length: int) -> List[int]:
    return [(x >> i) & 1 for i in range(length)]


def gf_tables(m: int, prim: int) -> Tuple[int, List[int], List[int]]:
    n = (1 << m) - 1
    alpha_to = [0] * (n + 1)
    index_of = [0] * (n + 1)

    mask = (1 << m) - 1
    alpha_to[0] = 1
    for i in range(1, n):
        a = alpha_to[i - 1] << 1
        if alpha_to[i - 1] & (1 << (m - 1)):
            a ^= prim
        alpha_to[i] = a & mask
    alpha_to[n] = 1

    index_of[0] = -1
    for i in range(n):
        index_of[alpha_to[i]] = i
    return n, alpha_to, index_of


def gf_mul(a: int, b: int, n: int, alpha_to: List[int], index_of: List[int]) -> int:
    if a == 0 or b == 0:
        return 0
    return alpha_to[(index_of[a] + index_of[b]) % n]


def gf_div(a: int, b: int, n: int, alpha_to: List[int], index_of: List[int]) -> int:
    if a == 0:
        return 0
    if b == 0:
        raise ZeroDivisionError("GF division by 0")
    return alpha_to[(index_of[a] - index_of[b]) % n]


def cyclotomic_coset(i: int, n: int) -> List[int]:
    s = set()
    x = i % n
    while x not in s:
        s.add(x)
        x = (x * 2) % n
    return sorted(s)


def poly_mul_field(p: List[int], q: List[int], n: int,
                   alpha_to: List[int], index_of: List[int]) -> List[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, pc in enumerate(p):
        if pc == 0:
            continue
        for j, qc in enumerate(q):
            if qc == 0:
                continue
            out[i + j] ^= gf_mul(pc, qc, n, alpha_to, index_of)
    return out


def poly_mul_bin(p: List[int], q: List[int]) -> List[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, pc in enumerate(p):
        if pc == 0:
            continue
        for j, qc in enumerate(q):
            if qc == 0:
                continue
            out[i + j] ^= 1
    return [c & 1 for c in out]


def minimal_polynomial(coset: List[int], n: int,
                       alpha_to: List[int], index_of: List[int]) -> List[int]:
    poly = [1]
    for e in coset:
        poly = poly_mul_field(poly, [alpha_to[e], 1], n, alpha_to, index_of)

    out = []
    for c in poly:
        if c == 0:
            out.append(0)
        elif c == 1:
            out.append(1)
        else:
            raise ValueError(f"Minimal polynomial coefficient not in GF(2): {c}")
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def bch_generator_poly(m: int, t: int) -> Tuple[int, int, int, List[int], List[int]]:
    prim = PRIM_POLY.get(m)
    if prim is None:
        raise ValueError(f"No primitive polynomial for m={m}")

    n, alpha_to, index_of = gf_tables(m, prim)

    used = set()
    cosets = []
    for i in range(1, 2 * t + 1):
        if i in used:
            continue
        cos = cyclotomic_coset(i, n)
        used.update(cos)
        cosets.append(cos)

    g_bits = [1]
    for cos in cosets:
        mp = minimal_polynomial(cos, n, alpha_to, index_of)
        g_bits = poly_mul_bin(g_bits, mp)

    g_int = 0
    for i, b in enumerate(g_bits):
        if b:
            g_int |= (1 << i)

    deg_g = len(g_bits) - 1
    return g_int, deg_g, n, alpha_to, index_of


def poly_mod_int(dividend: int, divisor: int) -> int:
    if divisor == 0:
        raise ZeroDivisionError("divisor=0")
    dd = dividend
    deg_div = divisor.bit_length() - 1
    while (dd.bit_length() - 1) >= deg_div:
        shift = (dd.bit_length() - 1) - deg_div
        dd ^= (divisor << shift)
    return dd


def bch_encode_shortened(data_bits: List[int], m: int, t: int) -> Tuple[List[int], Dict]:
    k_data = len(data_bits)
    g_int, deg_g, n, alpha_to, index_of = bch_generator_poly(m, t)
    k_full = n - deg_g
    if k_data > k_full:
        raise ValueError(f"data too long: k_data={k_data} > k_full={k_full} for (m={m}, t={t})")

    s = k_full - k_data
    parity_len = deg_g

    msg_int = bits_to_int(data_bits)
    shifted = msg_int << parity_len
    rem = poly_mod_int(shifted, g_int)
    code_int = shifted ^ rem

    code_bits_full = int_to_bits(code_int, n)
    tx_bits = code_bits_full[: n - s]

    meta = {
        "m": m, "t": t, "n": n,
        "deg_g": deg_g, "parity_len": parity_len,
        "k_full": k_full, "k_data": k_data,
        "shorten": s,
        "alpha_to": alpha_to, "index_of": index_of
    }
    return tx_bits, meta


def bch_decode_shortened(rx_bits: List[int], meta: Dict) -> Tuple[List[int], int]:
    m = meta["m"]
    t = meta["t"]
    n = meta["n"]
    deg_g = meta["deg_g"]
    parity_len = meta["parity_len"]
    k_full = meta["k_full"]
    k_data = meta["k_data"]
    s = meta["shorten"]
    alpha_to = meta["alpha_to"]
    index_of = meta["index_of"]
    field_n = (1 << m) - 1

    if len(rx_bits) != n - s:
        raise ValueError("rx length mismatch")

    r_full = rx_bits + [0] * s

    synd = [0] * (2 * t + 1)
    has_err = False
    for i in range(1, 2 * t + 1):
        s_i = 0
        for j, bit in enumerate(r_full):
            if bit:
                s_i ^= alpha_to[(i * j) % field_n]
        synd[i] = s_i
        if s_i != 0:
            has_err = True

    if not has_err:
        msg_bits = r_full[parity_len: parity_len + k_full]
        return msg_bits[:k_data], 0

    sigma = [0] * (t + 1)
    sigma[0] = 1
    B = [0] * (t + 1)
    B[0] = 1
    L = 0
    m_bm = 1
    b = 1

    for n_bm in range(1, 2 * t + 1):
        d = synd[n_bm]
        for i in range(1, L + 1):
            if sigma[i] != 0 and synd[n_bm - i] != 0:
                d ^= gf_mul(sigma[i], synd[n_bm - i], field_n, alpha_to, index_of)

        if d == 0:
            m_bm += 1
        else:
            T = sigma.copy()
            coef = gf_div(d, b, field_n, alpha_to, index_of)

            for i in range(0, t + 1 - m_bm):
                if B[i] != 0:
                    sigma[i + m_bm] ^= gf_mul(coef, B[i], field_n, alpha_to, index_of)

            if 2 * L <= n_bm - 1:
                L_new = n_bm - L
                B = T
                b = d
                L = L_new
                m_bm = 1
            else:
                m_bm += 1

    if L > t:
        raise RuntimeError(f"Too many errors: L={L} > t={t}")

    error_locs = []
    for j in range(field_n):
        x = alpha_to[(field_n - (j % field_n)) % field_n]
        val = 0
        x_pow = 1
        for i in range(0, L + 1):
            if sigma[i] != 0:
                val ^= gf_mul(sigma[i], x_pow, field_n, alpha_to, index_of)
            x_pow = gf_mul(x_pow, x, field_n, alpha_to, index_of)
        if val == 0:
            error_locs.append(j)

    if len(error_locs) != L:
        raise RuntimeError(
            f"Decoding failure: expected {L} roots, found {len(error_locs)}. "
            f"(errors may exceed t={t})"
        )

    corrected = r_full[:]
    for j in error_locs:
        corrected[j] ^= 1

    msg_bits = corrected[parity_len: parity_len + k_full]
    return msg_bits[:k_data], L


def pack_bits_to_bytes(bits_u8: np.ndarray) -> bytes:
    bits = np.asarray(bits_u8, dtype=np.uint8).flatten()
    if bits.size == 0:
        return b""
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be 0/1")
    pad = (-int(bits.size)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)], axis=0)
    b = bits.reshape(-1, 8)
    byte_vals = ((b[:, 0] << 7) | (b[:, 1] << 6) | (b[:, 2] << 5) | (b[:, 3] << 4) |
                 (b[:, 4] << 3) | (b[:, 5] << 2) | (b[:, 6] << 1) | (b[:, 7] << 0))
    return bytes(byte_vals.tolist())


def uint32_to_bits_msb(x: int) -> np.ndarray:
    x &= 0xFFFFFFFF
    return np.array([(x >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)


def bits_msb_to_uint32(bits32: np.ndarray) -> int:
    b = np.asarray(bits32, dtype=np.uint8).flatten()
    if b.size != 32:
        raise ValueError("bits32 length must be 32")
    v = 0
    for i in range(32):
        v = (v << 1) | int(b[i])
    return int(v)


def crc32_bits(payload_bits: np.ndarray) -> np.ndarray:
    data = pack_bits_to_bytes(payload_bits)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return uint32_to_bits_msb(crc)


def choose_params_for_errors(k_data: int, num_err: int,
                             min_m: int = 4, max_m: int = 15) -> Tuple[int, int]:
    t = max(1, int(num_err))
    for m in range(min_m, max_m + 1):
        _, deg_g, n, _, _ = bch_generator_poly(m, t)
        k_full = n - deg_g
        if k_data <= k_full:
            return m, t
    raise RuntimeError(f"Cannot find valid (m,t) for k_data={k_data}, num_err={num_err}.")


def choose_bch_params_for_error_rate(k_data: int, error_rate: float = 0.001,
                                     min_m: int = 4, max_m: int = 15,
                                     max_iter: int = 10) -> Tuple[int, int]:
    if k_data <= 0:
        raise ValueError("k_data must be positive")
    if not (0.0 <= error_rate <= 0.5):
        raise ValueError("error_rate must be in [0, 0.5]")

    t = max(1, int(math.ceil(error_rate * k_data)))
    for _ in range(max_iter):
        m, t = choose_params_for_errors(k_data, num_err=t, min_m=min_m, max_m=max_m)
        _, deg_g, n, _, _ = bch_generator_poly(m, t)
        k_full = n - deg_g
        s = k_full - k_data
        if s < 0:
            raise RuntimeError("Internal error: shortening negative")
        n_tx = n - s
        err_bound = int(math.ceil(error_rate * n_tx))
        if err_bound <= t:
            return m, t
        t = err_bound
    return m, t


def bch_ecc_encode_payload_single(payload_bits: np.ndarray, error_rate: float = 0.001,
                                  *, min_m: int = 4, max_m: int = 15) -> Tuple[np.ndarray, Dict]:
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).flatten()
    if np.any((payload_bits != 0) & (payload_bits != 1)):
        raise ValueError("payload_bits must be binary (0/1).")

    payload_k = int(payload_bits.size)

    if float(error_rate) <= 0.0:
        coded_bits = payload_bits.astype(np.uint8).copy()
        ecc_state = {
            "mode": "none",
            "error_rate": float(error_rate),
            "payload_k": payload_k,
            "crc_bits": 0,
            "k_data": payload_k,
            "m": 0,
            "t": 0,
            "max_err": 0,
            "n_tx": int(coded_bits.size),
            "bch_meta": None,
        }
        return coded_bits, ecc_state

    crc_bits = crc32_bits(payload_bits)
    msg_bits = np.concatenate([payload_bits, crc_bits], axis=0).astype(np.uint8)
    k_data = int(msg_bits.size)

    m, t = choose_bch_params_for_error_rate(k_data, error_rate=error_rate, min_m=min_m, max_m=max_m)
    tx_bits_list, bch_meta = bch_encode_shortened(msg_bits.tolist(), m, t)
    coded_bits = np.asarray(tx_bits_list, dtype=np.uint8)

    max_err = int(math.ceil(error_rate * coded_bits.size))
    ecc_state = {
        "mode": "single",
        "error_rate": float(error_rate),
        "payload_k": payload_k,
        "crc_bits": CRC_BITS,
        "k_data": k_data,
        "m": m,
        "t": t,
        "max_err": max_err,
        "n_tx": int(coded_bits.size),
        "bch_meta": bch_meta,
    }
    return coded_bits, ecc_state


def bch_ecc_encode_payload(payload_bits: np.ndarray, error_rate: float = 0.001,
                           *, block_k: int = 10000, min_m: int = 4, max_m: int = 15) -> Tuple[np.ndarray, Dict]:
    payload_bits = np.asarray(payload_bits, dtype=np.uint8).flatten()
    if np.any((payload_bits != 0) & (payload_bits != 1)):
        raise ValueError("payload_bits must be binary (0/1).")
    if block_k <= 0:
        raise ValueError("block_k must be positive")

    L = int(payload_bits.size)
    if L == 0:
        return np.zeros((0,), dtype=np.uint8), {"mode": "none", "payload_k": 0, "n_tx": 0, "error_rate": float(error_rate)}

    if float(error_rate) <= 0.0:
        coded_all = payload_bits.astype(np.uint8).copy()
        ecc_state = {
            "mode": "none",
            "error_rate": float(error_rate),
            "block_k": int(block_k),
            "payload_k": int(L),
            "n_tx": int(coded_all.size),
            "crc_bits": 0,
        }
        return coded_all, ecc_state

    if L <= block_k:
        return bch_ecc_encode_payload_single(payload_bits, error_rate=error_rate, min_m=min_m, max_m=max_m)

    blocks = []
    coded_parts = []
    off = 0
    while off < L:
        blk = payload_bits[off: off + block_k]
        coded_blk, st = bch_ecc_encode_payload_single(blk, error_rate=error_rate, min_m=min_m, max_m=max_m)
        blocks.append(st)
        coded_parts.append(coded_blk)
        off += int(blk.size)

    coded_all = np.concatenate(coded_parts, axis=0).astype(np.uint8)
    ecc_state = {
        "mode": "blockwise",
        "error_rate": float(error_rate),
        "block_k": int(block_k),
        "payload_k": int(L),
        "n_tx": int(coded_all.size),
        "blocks": blocks,
        "crc_bits": CRC_BITS,
    }
    return coded_all, ecc_state


def bch_ecc_decode_payload_single(extracted_coded_bits: np.ndarray, ecc_state: Dict) -> Tuple[np.ndarray, Dict]:
    extracted_coded_bits = np.asarray(extracted_coded_bits, dtype=np.uint8).flatten()
    if np.any((extracted_coded_bits != 0) & (extracted_coded_bits != 1)):
        raise ValueError("extracted_coded_bits must be binary (0/1).")

    mode = ecc_state.get("mode", "single")

    if mode == "none":
        payload_k = int(ecc_state["payload_k"])
        if int(extracted_coded_bits.size) != payload_k:
            return np.zeros((payload_k,), dtype=np.uint8), {
                "ok": False, "reason": "coded_length_mismatch",
                "expected_n_tx": payload_k, "got": int(extracted_coded_bits.size)
            }
        payload = extracted_coded_bits.astype(np.uint8).copy()
        info = {"ok": True, "bch_ok": True, "crc_ok": True, "corrected_num": 0, "t": 0, "max_err": 0}
        return payload, info

    n_tx = int(ecc_state["n_tx"])
    if int(extracted_coded_bits.size) != n_tx:
        return np.zeros((int(ecc_state["payload_k"]),), dtype=np.uint8), {
            "ok": False, "reason": "coded_length_mismatch", "expected_n_tx": n_tx,
            "got": int(extracted_coded_bits.size)
        }

    try:
        decoded_list, corrected_num = bch_decode_shortened(extracted_coded_bits.tolist(), ecc_state["bch_meta"])
        decoded = np.asarray(decoded_list, dtype=np.uint8)[: int(ecc_state["k_data"])]
        bch_ok = True
    except Exception as e:
        return np.zeros((int(ecc_state["payload_k"]),), dtype=np.uint8), {
            "ok": False, "reason": "bch_decode_failed", "error": str(e)
        }

    payload_k = int(ecc_state["payload_k"])
    crc_k = int(ecc_state.get("crc_bits", CRC_BITS))
    payload = decoded[:payload_k].astype(np.uint8)
    crc_rx = decoded[payload_k: payload_k + crc_k].astype(np.uint8)

    crc_calc = crc32_bits(payload)
    crc_ok = bool(np.array_equal(crc_rx, crc_calc))

    info = {
        "ok": bool(bch_ok and crc_ok),
        "bch_ok": bool(bch_ok),
        "crc_ok": bool(crc_ok),
        "corrected_num": int(corrected_num),
        "t": int(ecc_state["t"]),
        "max_err": int(ecc_state["max_err"]),
        "crc_rx": bits_msb_to_uint32(crc_rx),
        "crc_calc": bits_msb_to_uint32(crc_calc),
    }
    return payload, info


def bch_ecc_decode_payload(extracted_coded_bits: np.ndarray, ecc_state: Dict) -> Tuple[np.ndarray, Dict]:
    mode = ecc_state.get("mode", "single")
    extracted_coded_bits = np.asarray(extracted_coded_bits, dtype=np.uint8).flatten()

    if mode in ("single", "none"):
        return bch_ecc_decode_payload_single(extracted_coded_bits, ecc_state)

    if mode != "blockwise":
        raise ValueError(f"Unknown ecc_state mode: {mode}")

    blocks = ecc_state.get("blocks", [])
    if not blocks:
        return np.zeros((0,), dtype=np.uint8), {"ok": True, "num_blocks": 0, "blocks": []}

    expected = int(ecc_state["n_tx"])
    if int(extracted_coded_bits.size) != expected:
        return np.zeros((int(ecc_state["payload_k"]),), dtype=np.uint8), {
            "ok": False, "reason": "coded_length_mismatch", "expected_n_tx": expected,
            "got": int(extracted_coded_bits.size)
        }

    rec_parts = []
    infos = []
    off = 0
    ok_all = True
    corrected_total = 0

    for st in blocks:
        n_tx = int(st["n_tx"])
        chunk = extracted_coded_bits[off: off + n_tx]
        payload_blk, info_blk = bch_ecc_decode_payload_single(chunk, st)
        rec_parts.append(payload_blk)
        infos.append(info_blk)
        corrected_total += int(info_blk.get("corrected_num", 0))
        ok_all = ok_all and bool(info_blk.get("ok", False))
        off += n_tx

    recovered = np.concatenate(rec_parts, axis=0).astype(np.uint8)
    recovered = recovered[: int(ecc_state["payload_k"])]

    info = {
        "ok": bool(ok_all),
        "num_blocks": int(len(blocks)),
        "corrected_num_total": int(corrected_total),
        "blocks": infos,
    }
    return recovered, info


# =========================================================
# BCH+CRC wrapper for quad2bit RDH
# =========================================================

def embed_with_bch_ecc_rgb_even(cover_rgb_u8: np.ndarray,
                                payload_bits: np.ndarray,
                                error_rate: float = 0.001) -> Tuple[np.ndarray, Dict[str, int], int, Dict]:
    cover_rgb_u8 = clip_cover_rgb_u8(cover_rgb_u8)
    coded_bits, ecc_state = bch_ecc_encode_payload(payload_bits, error_rate=error_rate)

    stego_rgb_u8, embedded_per_channel, coded_len = embed_quad2bit_rgb_even(
        cover_rgb_u8=cover_rgb_u8,
        payload_bits=coded_bits
    )

    if coded_len != int(coded_bits.size):
        raise RuntimeError("Internal mismatch: coded_len != coded_bits.size")

    return stego_rgb_u8, embedded_per_channel, coded_len, ecc_state


def extract_with_bch_ecc_rgb_even(stego_rgb_u8: np.ndarray,
                                  cover_rgb_u8: np.ndarray,
                                  embedded_per_channel: Dict[str, int],
                                  coded_len: int,
                                  ecc_state: Dict) -> Tuple[np.ndarray, Dict]:
    cover_rgb_u8 = clip_cover_rgb_u8(cover_rgb_u8)
    extracted_coded_bits = extract_quad2bit_rgb_even(
        stego_rgb_u8=stego_rgb_u8,
        cover_rgb_u8=cover_rgb_u8,
        embedded_per_channel=embedded_per_channel,
        orig_len=int(coded_len)
    )
    recovered_payload, info = bch_ecc_decode_payload(extracted_coded_bits, ecc_state)
    info.update({"coded_len": int(coded_len), "n_tx": int(ecc_state.get("n_tx", -1))})
    return recovered_payload, info


# =========================================================
# Cross-environment utilities: deterministic payload + compact ECC metadata
# =========================================================
import json
import hashlib


def list_images(folder: str, exts=None):
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"folder does not exist or is not a directory: {folder}")
    out = []
    for fn in os.listdir(folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext in exts:
            out.append(fn)
    return sorted(out)


def payload_for_filename_run(filename: str, payload_seed: int, payload_len: int, run_idx: int = 1) -> np.ndarray:
    """
    Generate a deterministic payload using SHA-256.
    This avoids dependence on NumPy RNG implementation details across virtual environments.
    """
    payload_len = int(payload_len)
    if payload_len < 0:
        raise ValueError("payload_len must be >= 0")
    nbytes = (payload_len + 7) // 8
    material = f"payload_seed={int(payload_seed)}|run_idx={int(run_idx)}|filename={filename}|payload_len={payload_len}".encode("utf-8")
    buf = bytearray()
    counter = 0
    while len(buf) < nbytes:
        h = hashlib.sha256(material + b"|" + str(counter).encode("ascii")).digest()
        buf.extend(h)
        counter += 1
    if nbytes == 0:
        return np.zeros((0,), dtype=np.uint8)
    arr = np.frombuffer(bytes(buf[:nbytes]), dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")[:payload_len].astype(np.uint8)
    return bits


def bits_sha256(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be binary")
    packed = np.packbits(bits, bitorder="big").tobytes()
    return hashlib.sha256(f"bitlen={int(bits.size)}|".encode("ascii") + packed).hexdigest()


def _compact_bch_meta(meta: Optional[Dict]) -> Optional[Dict]:
    if meta is None:
        return None
    keys = ["m", "t", "n", "deg_g", "parity_len", "k_full", "k_data", "shorten"]
    return {k: int(meta[k]) for k in keys if k in meta}


def compact_ecc_state(ecc_state: Dict) -> Dict:
    """
    Convert ecc_state to JSON-serializable metadata.
    alpha_to/index_of are intentionally not stored; they are reconstructed in the extraction environment.
    """
    mode = ecc_state.get("mode", "single")
    out = {
        "mode": mode,
        "error_rate": float(ecc_state.get("error_rate", 0.0)),
        "payload_k": int(ecc_state.get("payload_k", 0)),
        "n_tx": int(ecc_state.get("n_tx", 0)),
        "crc_bits": int(ecc_state.get("crc_bits", 0)),
    }
    if mode == "blockwise":
        out["block_k"] = int(ecc_state.get("block_k", 0))
        out["blocks"] = [compact_ecc_state(st) for st in ecc_state.get("blocks", [])]
    elif mode == "single":
        for k in ["k_data", "m", "t", "max_err"]:
            out[k] = int(ecc_state[k])
        out["bch_meta"] = _compact_bch_meta(ecc_state.get("bch_meta"))
    elif mode == "none":
        for k in ["block_k", "k_data", "m", "t", "max_err"]:
            if k in ecc_state:
                out[k] = int(ecc_state[k])
        out["bch_meta"] = None
    else:
        raise ValueError(f"Unsupported ecc_state mode: {mode}")
    return out


def _restore_bch_meta(meta: Optional[Dict]) -> Optional[Dict]:
    if meta is None:
        return None
    m = int(meta["m"])
    prim = PRIM_POLY.get(m)
    if prim is None:
        raise ValueError(f"No primitive polynomial for m={m}")
    n, alpha_to, index_of = gf_tables(m, prim)
    out = {k: int(meta[k]) for k in ["m", "t", "n", "deg_g", "parity_len", "k_full", "k_data", "shorten"]}
    if int(out["n"]) != int(n):
        raise ValueError(f"BCH metadata n mismatch: metadata n={out['n']}, reconstructed n={n}")
    out["alpha_to"] = alpha_to
    out["index_of"] = index_of
    return out


def restore_ecc_state(compact: Dict) -> Dict:
    mode = compact.get("mode", "single")
    out = {
        "mode": mode,
        "error_rate": float(compact.get("error_rate", 0.0)),
        "payload_k": int(compact.get("payload_k", 0)),
        "n_tx": int(compact.get("n_tx", 0)),
        "crc_bits": int(compact.get("crc_bits", 0)),
    }
    if mode == "blockwise":
        out["block_k"] = int(compact.get("block_k", 0))
        out["blocks"] = [restore_ecc_state(st) for st in compact.get("blocks", [])]
    elif mode == "single":
        for k in ["k_data", "m", "t", "max_err"]:
            out[k] = int(compact[k])
        out["bch_meta"] = _restore_bch_meta(compact.get("bch_meta"))
    elif mode == "none":
        for k in ["block_k", "k_data", "m", "t", "max_err"]:
            if k in compact:
                out[k] = int(compact[k])
        out["bch_meta"] = None
    else:
        raise ValueError(f"Unsupported ECC mode: {mode}")
    return out


def error_rate_tag(error_rate: float) -> str:
    return f"er_{float(error_rate):.10f}"


def run_tag(run_idx: int) -> str:
    return f"run_{int(run_idx):03d}"

# =========================================================
# Extraction-side program
# =========================================================


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}")
    return records


def extract_one_run(
    cover_extract_dir: str,
    run_dir: str,
    summary_csv_path: Optional[str] = None,
    stop_on_failed_images_per_run: int = 2,
) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_jsonl(metadata_path)
    if not records:
        raise ValueError(f"No metadata records in {metadata_path}")
    if int(stop_on_failed_images_per_run) < 1:
        raise ValueError("stop_on_failed_images_per_run 必须 >= 1。")

    if summary_csv_path is None:
        summary_csv = run_dir / "extract_summary.csv"
    else:
        summary_csv = Path(summary_csv_path)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    ok_cnt = 0
    err_cnt = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as csv_f:
        w = csv.writer(csv_f)
        w.writerow([
            "filename", "status", "H", "W", "payload_len", "payload_seed", "run_idx", "error_rate",
            "payload_sha256", "payload_hash_match", "coded_len", "per_len_bits",
            "bit_errors", "ber_payload", "exact_match", "corrected_num_total", "crc_all_ok", "note"
        ])

        for rec in records:
            fn = rec["filename"]

            # Cross-platform path normalization:
            # metadata.jsonl may contain Windows-style paths such as "stego\\2_0001.png".
            # On Linux/macOS, backslash is not treated as a path separator, so normalize it to "/".
            rel = rec.get("stego_relpath", f"stego/{fn}")
            rel = str(rel).replace("\\", "/").lstrip("/")
            stego_path = run_dir / Path(rel)

            cover_path = Path(cover_extract_dir) / fn
            note = ""
            try:
                if not stego_path.exists():
                    raise FileNotFoundError(f"missing stego: {stego_path}")
                if not cover_path.exists():
                    raise FileNotFoundError(f"missing extraction cover: {cover_path}")

                payload_len = int(rec.get("payload_len", manifest.get("payload_len")))
                payload_seed = int(rec.get("payload_seed", manifest.get("payload_seed")))
                run_idx = int(rec.get("run_idx", manifest.get("run_idx", 1)))
                error_rate = float(rec.get("error_rate", manifest.get("error_rate")))
                payload_gt = payload_for_filename_run(fn, payload_seed, payload_len, run_idx)
                payload_hash = bits_sha256(payload_gt)
                payload_hash_match = int(payload_hash == rec.get("payload_sha256", ""))
                if not payload_hash_match:
                    note += "payload_hash_mismatch;"

                stego_rgb = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)
                cover_rgb = np.array(Image.open(cover_path).convert("RGB"), dtype=np.uint8)

                emb_per_ch = {k: int(v) for k, v in rec["embedded_per_channel"].items()}
                coded_len = int(rec["coded_len"])
                ecc_state = restore_ecc_state(rec["ecc_state"])

                recovered, info = extract_with_bch_ecc_rgb_even(
                    stego_rgb_u8=stego_rgb,
                    cover_rgb_u8=cover_rgb,
                    embedded_per_channel=emb_per_ch,
                    coded_len=coded_len,
                    ecc_state=ecc_state,
                )
                recovered = np.asarray(recovered, dtype=np.uint8).flatten()[:payload_len]
                payload_flat = np.asarray(payload_gt, dtype=np.uint8).flatten()[:payload_len]

                if recovered.size != payload_flat.size:
                    raise RuntimeError(f"recovered length mismatch: {recovered.size} vs {payload_flat.size}")

                bit_errors = int(np.sum(recovered != payload_flat, dtype=np.uint64))
                ber = bit_errors / float(payload_len) if payload_len > 0 else 0.0
                exact = bool(bit_errors == 0)
                corrected_total = int(info.get("corrected_num_total", info.get("corrected_num", 0)))
                crc_all_ok = bool(info.get("ok", False)) if "ok" in info else bool(info.get("crc_ok", False))
                status = "ok" if exact else "error:PayloadMismatch"
                if not crc_all_ok:
                    note += "crc_or_bch_not_ok;"

                H, W_img = int(stego_rgb.shape[0]), int(stego_rgb.shape[1])
                per_len_bits = int(emb_per_ch["R"])
                w.writerow([
                    fn, status, H, W_img, payload_len, payload_seed, run_idx, f"{error_rate:.10f}",
                    payload_hash, payload_hash_match, coded_len, per_len_bits,
                    bit_errors, ber, int(exact), corrected_total, int(crc_all_ok), note
                ])
                csv_f.flush()

                if exact:
                    ok_cnt += 1
                    print(f"[EXTRACT OK] {fn} coded_len={coded_len} ber={ber:.3g} corrected={corrected_total} crc={int(crc_all_ok)}")
                else:
                    err_cnt += 1
                    print(f"[EXTRACT FAIL] {fn} coded_len={coded_len} ber={ber:.6g} bit_errors={bit_errors} corrected={corrected_total} crc={int(crc_all_ok)}")
                    if err_cnt >= int(stop_on_failed_images_per_run):
                        print(
                            f"[STOP CURRENT RUN] 失败图像数已达到 {err_cnt}，"
                            f"触发 stop_on_failed_images_per_run={int(stop_on_failed_images_per_run)}，停止当前 run 的提取。"
                        )
                        break
            except Exception as e:
                err_cnt += 1
                w.writerow([
                    fn, f"error:{type(e).__name__}", "", "", rec.get("payload_len", ""), rec.get("payload_seed", ""),
                    rec.get("run_idx", ""), rec.get("error_rate", ""), rec.get("payload_sha256", ""), "",
                    rec.get("coded_len", ""), "", "", "", "", "", "", str(e)
                ])
                csv_f.flush()
                print(f"[EXTRACT ERR] {fn} -> {type(e).__name__}: {e}")
                if err_cnt >= int(stop_on_failed_images_per_run):
                    print(
                        f"[STOP CURRENT RUN] 失败图像数已达到 {err_cnt}，"
                        f"触发 stop_on_failed_images_per_run={int(stop_on_failed_images_per_run)}，停止当前 run 的提取。"
                    )
                    break

    print(f"[EXTRACT DONE] run_dir={run_dir} ok={ok_cnt} err={err_cnt}")
    print(f"[SUMMARY] {summary_csv}")
    return summary_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction side for quad-2bit RDH with BCH+CRC. Reads run-specific stego files and metadata, then computes payload BER.")
    parser.add_argument("--cover_extract_dir", type=str, default="D:/code/IS/Key_DDIM/result2/ddim2_batch1000_env2_g1.0_0_3_3_10000", help="Extraction-side cover directory")
    parser.add_argument("--run_dir", type=str, default="D:/code/IS/LX/result2/quad_bch_runs/er_0.0006000000/run_001", help="Run directory generated by rdh_quad_bch_embed.py, e.g. ./result/quad_bch_runs/er_0.0069000000/run_001")
    parser.add_argument("--summary_csv", type=str, default=None, help="Where to save extraction summary CSV. Default: <run_dir>/extract_summary.csv")
    parser.add_argument("--stop_on_failed_images_per_run", type=int, default=1,
                        help="当前 run 中允许失败的图像数阈值。默认2，表示失败图像达到2张时停止该 run 的提取。")
    args = parser.parse_args()

    extract_one_run(
        cover_extract_dir=args.cover_extract_dir,
        run_dir=args.run_dir,
        summary_csv_path=args.summary_csv,
        stop_on_failed_images_per_run=int(args.stop_on_failed_images_per_run),
    )
