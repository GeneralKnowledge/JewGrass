"""Glen Gothic palette (screen-print inks). Values are sRGB 0-255.

Warning red is reserved for money and ownership: boundary lines, the fund's
logo, price tags, stamps. Nothing else is red.
"""

PEAT = (24, 21, 22)
BONE = (233, 225, 206)
HEATHER = (112, 86, 120)
LOCH = (46, 84, 92)
FOG = (168, 170, 166)
MOSS = (78, 86, 62)
WARNING_RED = (196, 38, 34)


def f(rgb):
    """sRGB tuple -> float 0..1 tuple."""
    return tuple(c / 255.0 for c in rgb)


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))
