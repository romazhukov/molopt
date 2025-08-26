import re

def parse_total_energy(text: str):
    """
    Пытается найти строку с энергией в Eh в xtb логах/выводе.
    """
    m = re.search(r"TOTAL ENERGY\s*[-=:\s]*([-\d\.Ee+]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None
