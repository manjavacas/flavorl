import re
import pandas as pd
import ast

# --- parser de duraciones -> minutos
_TIME_PAT = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)\b",
    flags=re.I,
)


def _to_minutes(s: str) -> int:
    if not s:
        return None
    total = 0.0
    for m in _TIME_PAT.finditer(s):
        num = float(m.group("num").replace(",", "."))
        unit = m.group("unit").lower()
        if unit in ("h", "hr", "hrs", "hour", "hours"):
            total += num * 60
        elif unit in ("m", "min", "mins", "minute", "minutes"):
            total += num
        elif unit in ("s", "sec", "secs", "second", "seconds"):
            total += num / 60.0
    return int(round(total)) if total > 0 else None


# --- extrae Prep / Cook / Ready In y limpia el texto
_HDR_PAT = re.compile(r"\b(Prep|Cook|Ready\s*In)\b\s*[:\-]?\s*([^\n\r]+)", flags=re.I)


def extract_times_and_clean(text: str):
    """
    Entrada: string de 'directions'/'cooking_directions' que arranca con:
      Prep\n20 m\nCook\n1 h\nReady In\n1 h 40 m\n...
    o variantes en una sola línea.
    Salida: (prep_min, cook_min, ready_min, cleaned_text)
    """
    if text is None:
        return (None, None, None, None)

    s = str(text)
    matches = list(_HDR_PAT.finditer(s))
    # Tomamos la primera ocurrencia de cada etiqueta en orden de aparición
    prep_val = cook_val = ready_val = None
    end_cut = 0
    seen = set()
    for m in matches:
        label = m.group(1).lower().replace(" ", "")  # 'readyin' para "Ready In"
        val = m.group(2).strip()
        if label == "prep" and "prep" not in seen:
            prep_val = val
            seen.add("prep")
            end_cut = max(end_cut, m.end())
        elif label == "cook" and "cook" not in seen:
            cook_val = val
            seen.add("cook")
            end_cut = max(end_cut, m.end())
        elif label == "readyin" and "readyin" not in seen:
            ready_val = val
            seen.add("readyin")
            end_cut = max(end_cut, m.end())
        # paramos cuando ya tenemos las 3
        if len(seen) == 3:
            break

    # Si el patrón viene en líneas intercaladas (etiqueta en una línea y valor en la siguiente),
    # el regex anterior también lo captura porque toma hasta el fin de línea. Si no encontró alguno, intentamos
    # recuperar mirando el bloque inicial (las 6 primeras líneas).
    if len(seen) < 3:
        lines = s.splitlines()
        head = " ".join(lines[:8])  # vistazo corto
        if prep_val is None:
            m = re.search(r"\bPrep\b\s*[:\-]?\s*([^\n\r]+)", head, flags=re.I)
            if m:
                prep_val = m.group(1).strip()
        if cook_val is None:
            m = re.search(r"\bCook\b\s*[:\-]?\s*([^\n\r]+)", head, flags=re.I)
            if m:
                cook_val = m.group(1).strip()
        if ready_val is None:
            m = re.search(r"\bReady\s*In\b\s*[:\-]?\s*([^\n\r]+)", head, flags=re.I)
            if m:
                ready_val = m.group(1).strip()

    # Normalizamos a minutos
    prep_min = _to_minutes(prep_val)
    cook_min = _to_minutes(cook_val)
    ready_min = _to_minutes(ready_val)

    # Texto limpio: desde el final del tercer match si lo hubo; si no, intentamos
    # cortar las primeras ~6 líneas típicas (etiqueta/valor x3)
    if end_cut > 0:
        cleaned = s[end_cut:].lstrip(" \n\r\t:.-")
    else:
        # fallback: quita la cabecera si claramente son 6 líneas 'label/valor'
        lines = s.splitlines()
        cleaned = "\n".join(lines[6:]).lstrip() if len(lines) >= 6 else s

    # Si después de limpiar queda vacío o solo espacios, usar el texto original
    if not cleaned.strip():
        cleaned = s

    return (prep_min, cook_min, ready_min, cleaned)


def _clean_and_unpack(x):
    # 1) Parsear a dict si viene como string "{'directions': u'...'}"
    d = x
    if isinstance(x, str):
        try:
            d = ast.literal_eval(x.strip())
        except Exception:
            d = {"directions": x}
    if not isinstance(d, dict):
        d = {"directions": str(x)}
    # 2) Aplicar tu función al texto
    prep, cook, ready, cleaned = extract_times_and_clean(d.get("directions"))
    # 3) Mantener el mismo formato (dict) pero con 'directions' limpio
    d_out = dict(d)
    d_out["directions"] = cleaned
    # 4) Devolver TODO como Series (4 campos)
    return pd.Series(
        {
            "cooking_directions": d_out,
            "prep_min": prep,
            "cook_min": cook,
            "ready_min": ready,
        }
    )


# sample = u'Prep\n20 m\nCook\n1 h\nReady In\n1 h 40 m\nGrease and flour two 8 x 4 inch pans. Preheat oven to 325 degrees F (165 degrees C).\nSift flour, salt, baking powder, soda, and cinnamon together in a bowl.\nBeat eggs, oil, vanilla, and sugar together in a large bowl. Add sifted ingredients to the creamed mixture, and beat well. Stir in zucchini and nuts until well combined. Pour batter into prepared pans.\nBake for 40 to 60 minutes, or until tester inserted in the center comes out clean. Cool in pan on rack for 20 minutes. Remove bread from pan, and completely cool.'
# prep, cook, ready, cleaned = extract_times_and_clean(sample)
# print(prep, cook, ready)     # 20, 60, 100
# print(cleaned[:80], "...")
