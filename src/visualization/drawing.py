"""Pump station elevation drawing: only what the model uses, elevations to scale.

Run from the repository root: python -m src.visualization.drawing [light|dark]
The PNG is rendered from the SVG with: rsvg-convert -z 2 docs/plant.svg -o docs/plant.png
"""
W, H = 1200, 740
import sys
THEME = sys.argv[1] if len(sys.argv) > 1 else "light"  # run once per theme
PALETTES = {
    "light": dict(bg="#F5F5F7", ink="#1D1D1F", line="#8A8A8F", sec="#6E6E73", hair="#C7C7CC", arrow="#2C74C0", water="#CDE4F5"),
    "dark": dict(bg="#242427", ink="#F5F5F7", line="#7C7C82", sec="#A1A1A6", hair="#48484A", arrow="#6AAEF5", water="#1D3A5C"),
}
P = PALETTES[THEME]
BG, INK, SEC, HAIR = P["bg"], P["ink"], P["sec"], P["hair"]
LINE = P["line"]  # every drawn line sits a step below the text, as in Apple's label hierarchy
JADE = {"base": P["arrow"], "water": P["water"]}
PIPE = 20  # outside diameter on the drawing, px
S = 32
def Y(el): return 590 - S * el

o = []; a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter, SF Pro Text, Helvetica Neue, Arial">')
a(f'<rect width="{W}" height="{H}" rx="24" fill="{BG}"/>')
def line(x1, y1, x2, y2, c=LINE, w=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}"{d} stroke-linecap="round"/>')
def text(x, y, s, size=11, c=SEC, anchor="start", weight=400, rot=None):
    t = f' transform="rotate({rot} {x} {y})"' if rot else ''
    a(f'<text x="{x}" y="{y}" font-size="{size}" fill="{c}" text-anchor="{anchor}" font-weight="{weight}"{t}>{s}</text>')
def bubble(x, y, tag, num="101"):
    a(f'<circle cx="{x}" cy="{y}" r="18" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
    text(x, y - 1, tag, 10.5, INK, "middle", 600); text(x, y + 11, num, 9.5, INK, "middle")
def level(x, y, label):
    a(f'<polygon points="{x-6},{y-11} {x+6},{y-11} {x},{y-1}" fill="none" stroke="{JADE["base"]}" stroke-width="1.1"/>')
    text(x + 12, y - 3, label, 10.5, INK)
def vdim(x, y1, y2, label):
    line(x, y1, x, y2, LINE, 0.8)
    for y, s in ((y1, 1), (y2, -1)): a(f'<polygon points="{x-3.5},{y+s*9} {x+3.5},{y+s*9} {x},{y}" fill="{LINE}"/>')
    text(x - 7, (y1 + y2) / 2, label, 10.5, INK, "middle", rot=-90)
def tank(x1, x2, bottom, top, water, name, mark_el):
    a(f'<rect x="{x1}" y="{Y(water)}" width="{x2-x1}" height="{bottom-Y(water)}" fill="{JADE["water"]}"/>')
    line(x1, Y(water), x2, Y(water), JADE["base"], 1.2)
    a(f'<polyline points="{x1},{top} {x1},{bottom} {x2},{bottom} {x2},{top}" fill="none" stroke="{LINE}" stroke-width="1.6"/>')
    level(x1 + 20, Y(water), mark_el)
    text((x1 + x2) / 2, bottom - 14, name, 11, INK, "middle", 500)

text(W / 2, 54, "Pump Station", 27, INK, "middle", 600)
text(W / 2, 80, "Centrifugal pump lifting water to a reservoir", 13, SEC, "middle")

floor = Y(-1.0)
line(90, floor, 1110, floor, LINE, 1.2)
text(920, floor - 8, "Floor EL −1.00", 10.5, SEC, "end")

# tanks
tank(100, 240, floor, Y(2.6), 2.0, "Suction tank", "EL +2.00")
rbot = Y(11)
tank(940, 1080, rbot, Y(15), 14.0, "Reservoir", "EL +14.00")
for x in (960, 1060): line(x, rbot, x, floor, LINE, 1.4)

# process lines: pipe walls in ink, water inside, flow arrows in the water
yc, dx, top = Y(0), 598, Y(12)
def pipe(points, start_trim=0, end_trim=0):
    pts = " ".join(f"{x},{y}" for x, y in points)
    a(f'<polyline points="{pts}" fill="none" stroke="{LINE}" stroke-width="{PIPE}" stroke-linejoin="miter"/>')
    inner = list(points)
    (x0, y0), (x1, y1) = inner[0], inner[-1]
    inner[0] = (x0 - start_trim, y0); inner[-1] = (x1 + end_trim, y1)
    pts = " ".join(f"{x},{y}" for x, y in inner)
    a(f'<polyline points="{pts}" fill="none" stroke="{JADE["water"]}" stroke-width="{PIPE-3.2}" stroke-linejoin="miter"/>')
pipe([(240, yc), (590, yc)], start_trim=2)
pipe([(dx, yc), (dx, top), (940, top)], end_trim=2)
# one arrow per run, each centred on its visible length
sx = (240 + 566) / 2          # tank wall to pump casing
ry = ((yc - 24) + top) / 2    # pump casing to elbow
for p in (f"{sx-5.5},{yc-5} {sx+5.5},{yc} {sx-5.5},{yc+5}", f"{dx-5},{ry+5.5} {dx},{ry-5.5} {dx+5},{ry+5.5}"):
    a(f'<polygon points="{p}" fill="{JADE["base"]}" fill-opacity="0.55"/>')

# pump, motor, drive
a(f'<circle cx="590" cy="{yc}" r="24" fill="{BG}" stroke="{LINE}" stroke-width="1.6"/>')
text(556, yc - 18, "℄ EL 0.00", 10.5, SEC, "end")
line(614, yc, 656, yc, LINE, 2)
a(f'<circle cx="680" cy="{yc}" r="24" fill="{BG}" stroke="{LINE}" stroke-width="1.6"/>')
text(680, yc + 5, "M", 14, INK, "middle", 500)
line(704, yc, 750, yc, LINE, 1.2)
a(f'<rect x="750" y="{yc-16}" width="80" height="32" rx="6" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
text(790, yc + 4, "VFD", 11, INK, "middle", 500)

# sensors
row = 495
line(212, Y(2.6) - 12, 212, Y(1.0), LINE, 0.9); bubble(212, Y(2.6) - 30, "LT")
for x, tag in ((380, "TT"), (450, "PT")):
    line(x, row + 18, x, yc - PIPE / 2, LINE, 0.9); bubble(x, row, tag)
line(578, row, dx - PIPE / 2, row, LINE, 0.9); bubble(560, row, "PT", "102")
for x, tag in ((760, "ST"), (820, "JT")):
    line(x, row + 18, x, yc - 16, LINE, 0.9, "4 3"); bubble(x, row, tag)
a(f'<rect x="748" y="{top-13}" width="36" height="26" rx="6" fill="{BG}" stroke="{LINE}" stroke-width="1.2"/>')
line(766, top - 13, 766, top - 30, LINE, 0.9); bubble(766, top - 48, "FT")

# heads
line(240, Y(2.0), 890, Y(2.0), HAIR, 0.9, "6 4")
line(870, Y(14), 940, Y(14), HAIR, 0.9, "6 4")
vdim(880, Y(14), Y(2.0), "Static head 12.00 m")
vdim(262, Y(2.0), yc, "2.00 m")

# specification row
cells = [("Pump", "End suction 3×2–10, 1750 rpm"), ("Duty point", "57 m³/h at 18.9 m"),
         ("Fluid", "Water, 20 °C"), ("Piping", "DN100")]
cw = 1020 / len(cells)
for i, (k, v) in enumerate(cells):
    x = 100 + i * cw
    text(x, 680, k, 10.5, SEC); text(x, 701, v, 13, INK, weight=500)
a('</svg>')
open(f"docs/plant{'' if THEME == 'light' else '-dark'}.svg", "w").write("\n".join(o))
