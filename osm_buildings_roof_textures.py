"""
osm_buildings_roof_textures.py
(Roof-texture version of osm_buildings_on_terrain.py: same terrain + buildings, plus
NAIP roof textures aligned to OSM. Writes to its own output_*roof_textures* folders.)

Combine OpenStreetMap building footprints (extruded to 3D) with a terrain mesh,
and drape USGS NAIP aerial imagery on the terrain as a ground texture.
Default test area: University of Michigan North Campus, centered on Pierpont Commons.

Terrain sources
  terrarium : Mapzen/Tilezen Terrain Tiles on AWS Open Data (the same DEM source
              Mapscad uses). No API key. In the US these are largely derived from
              USGS 3DEP/NED. Zoom 15 ~= 3.5 m/pixel at this latitude.
  3dep      : USGS 3DEP directly via py3dep (optional, higher resolution where
              lidar exists). Requires: pip install py3dep

Ground texture (NAIP)
  default   : USGS The National Map NAIP ImageServer (USGSNAIPImagery), the same
              NAIP holdings EROS distributes. No API key. Newest year by default;
              --naip-year 2022 locks a specific year. Requested directly in the
              scene's UTM zone so pixels line up with the mesh grid.
  local     : --naip-file one or more GeoTIFF / JP2 tiles you downloaded from
              EarthExplorer (EROS archive) or The National Map Downloader.
              Requires: pip install rasterio
  off       : --no-naip keeps the old height-based terrain colors.

Roof textures (--roof-texture)
  OSM footprints and the roofs visible in NAIP do not line up: NAIP georegistration
  error, OSM tracing error, and relief displacement (roofs lean away from the camera's
  nadir) shift each roof by a few meters. The 3D buildings stay on the OSM footprints;
  only the roof UVs are shifted so each roof samples the image where that roof really is.
    --roof-align global  one offset for the whole AOI (default). Found by matching
                         footprint outlines to image edges (x/y-oriented, FFT search),
                         the same idea as 02_align_osm_to_raster.py with the y sign fixed.
    --roof-align local   global, then a small per-building search around it; a building
                         keeps its own offset only when the match is clearly better.
    --roof-offset "dx, dy"  manual offset in meters (east, north), skips detection.
  With roof textures on, the ground drape is inpainted where roofs appear in the image,
  so no "ghost" roof shows on the ground next to a building (--no-ground-inpaint: off).

Output folder (created where you run the command; --out to choose your own)
  output_roof_textures/                --roof-texture                       (normal use)
  output_roof_textures_per_building/   --roof-texture --roof-align local
  output_roof_textures_manual_shift/   --roof-texture --roof-offset "dx, dy"
  output_roof_textures_no_shift/       --roof-texture --roof-align off      (for comparison)
  output_no_roof_textures/             without --roof-texture (gray buildings)
  None of these is output/, so runs of osm_buildings_on_terrain.py are never overwritten.

Outputs (in that folder):
  north_campus_scene.glb           textured scene, Y-up (glTF) -> Blender, three.js
  north_campus_terrain_naip.obj    terrain only, Z-up, with .mtl + texture
  north_campus_combined.obj        single mesh, Z-up, meters (vertex colors, no texture)
  north_campus_combined.stl        single mesh, Z-up, meters (scale in your slicer)
  naip_texture.jpg / .jgw          the draped image + world file (UTM) for GIS checks
  north_campus_google_earth.kmz    open in Google Earth Pro: 3D terrain + buildings at
                                   their real location, NAIP / hillshade overlays, AOI outline
  buildings_footprints_utm.geojson footprints actually used (UTM), with height + height source
  terrain_hillshade.png            hillshade of the terrain grid, same extent as the NAIP image
  naip_hillshade_blend.jpg         NAIP and hillshade blended, to eyeball image vs. terrain
  terrain_grid.npz                 terrain heights (UTM grid, m) for check_terrain_fidelity.py
  roof_alignment_preview.jpg       (--roof-texture) NAIP with OSM outlines: red = raw,
                                   cyan = where the roof texture is taken from
  naip_ground_inpainted.jpg        (--roof-texture) ground drape with roofs removed
  (then run check_terrain_fidelity.py on this folder: terrain vs USGS 3DEP 1 m lidar,
   NAIP vs ground alignment; check_naip_offsets.py: NAIP vs OSM footprints)

Install:
  pip install osmnx trimesh numpy scipy pyproj requests pillow shapely mapbox-earcut
  (optional) pip install py3dep rasterio

Run (VS Code terminal):
  python osm_buildings_roof_textures.py                      # asks for coordinates (Enter = North Campus)
  python osm_buildings_roof_textures.py --center "42.2917, -83.7174" --size 1200
  python osm_buildings_roof_textures.py --naip-year 2022 --roof-texture
  python osm_buildings_roof_textures.py --roof-texture --roof-align local
  python osm_buildings_roof_textures.py --roof-texture --roof-offset "5.4, -0.6"
  python osm_buildings_roof_textures.py --naip-file m_4208357_ne_17_060_20220614.tif
  python osm_buildings_roof_textures.py --bbox "-83.7250, 42.2860, -83.7090, 42.2990"
  python osm_buildings_roof_textures.py --preset north_campus --solid
  python osm_buildings_roof_textures.py --source 3dep --dem-res 1
  python osm_buildings_roof_textures.py --kmz-altitude anchor   # pin model to Google's ground at center

Coordinate formats
  --center "lat, lon"               e.g. "42.2917, -83.7174"  (Google Maps right-click order)
  --bbox   "west, south, east, north" in degrees (lon/lat order, like OSM / GeoJSON)
"""

import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

try:
    import numpy as np
    import requests
    import trimesh
    from PIL import Image
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import (binary_dilation, distance_transform_edt, gaussian_filter,
                               map_coordinates, sobel)
    from shapely.geometry import MultiPolygon, Polygon, box
    from shapely.affinity import translate as shp_translate
except ModuleNotFoundError as _exc:
    sys.exit(
        f"\nMissing package: {_exc.name}\n"
        f"This script is running with: {sys.executable}\n"
        f"Install into THIS interpreter with:\n"
        f"  {sys.executable} -m pip install osmnx trimesh numpy scipy pyproj "
        f"requests pillow shapely mapbox-earcut\n"
        f"or run the script with the Python where you installed them "
        f"(e.g. /opt/anaconda3/bin/python osm_buildings_roof_textures.py).\n"
    )

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
CENTER_LAT = 42.2917      # Pierpont Commons, UMich North Campus
CENTER_LON = -83.7174
UTM_CRS = "EPSG:32617"    # UTM zone 17N (Ann Arbor), units = meters
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
NAIP_URL = ("https://imagery.nationalmap.gov/arcgis/rest/services/"
            "USGSNAIPImagery/ImageServer/exportImage")
NAIP_MAX_PX = 4000        # service maxImageWidth/Height; larger requests are tiled

LEVEL_HEIGHT_M = 3.2      # meters per building level when no height tag
DEFAULT_HEIGHT_M = 8.0    # fallback when neither height nor levels are tagged
EMBED_M = 0.1             # foundation extends this far below the lowest ground point

TERRAIN_COLOR_LOW = np.array([110, 140, 90])
TERRAIN_COLOR_HIGH = np.array([170, 155, 120])
BUILDING_COLOR = [215, 215, 220, 255]


# ----------------------------------------------------------------------------
# Coordinate input
# ----------------------------------------------------------------------------
# Example areas. Center = Pierpont Commons; bbox is an approximate outline of
# UMich North Campus (Plymouth Rd to Hubbard Rd, Huron Pkwy to Fuller/Bonisteel).
PRESETS = {
    "north_campus": {
        "center": (42.2917, -83.7174),                      # (lat, lon)
        "size": 1200,                                       # meters
        "bbox": (-83.7250, 42.2860, -83.7090, 42.2990),     # (west, south, east, north)
    },
    "pierpont": {
        "center": (42.2917, -83.7174),
        "size": 500,
    },
}


def _numbers(text):
    vals = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    return vals


def parse_center(text):
    """'42.2917, -83.7174' -> (lat, lon). Validates ranges."""
    vals = _numbers(text)
    if len(vals) != 2:
        raise ValueError(f"expected 'lat, lon', got: {text!r}")
    lat, lon = vals
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"out of range lat/lon: {lat}, {lon}")
    if abs(lat) > 80 and abs(lon) <= 80:
        print("[warn] lat looks like a longitude — did you swap the order? (expected lat, lon)")
    return lat, lon


def parse_bbox(text):
    """'west, south, east, north' -> tuple. Validates ordering."""
    vals = _numbers(text)
    if len(vals) != 4:
        raise ValueError(f"expected 'west, south, east, north', got: {text!r}")
    w, s, e, n = vals
    if not (w < e and s < n):
        raise ValueError("bbox must satisfy west < east and south < north")
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        raise ValueError("bbox values out of range (lon -180..180, lat -90..90)")
    return w, s, e, n


def bbox_to_center_size(bbox):
    """Center (lat, lon) and the edge length (m) of a square that covers the bbox."""
    w, s, e, n = bbox
    lat, lon = (s + n) / 2, (w + e) / 2
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs_for(lat, lon), always_xy=True)
    xs, ys = to_utm.transform([w, e, e, w], [s, s, n, n])
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    return (lat, lon), size


def utm_crs_for(lat, lon):
    """Pick the UTM zone for any location (Ann Arbor -> EPSG:32617)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def prompt_for_area():
    """Interactive input. Press Enter to use the North Campus example."""
    ex = PRESETS["north_campus"]
    print("\nChoose the area (press Enter to use the example):")
    print(f"  1) center point + size   example: {ex['center'][0]}, {ex['center'][1]}  / {ex['size']} m")
    print(f"  2) bounding box          example: {', '.join(map(str, ex['bbox']))}  (W, S, E, N)")
    mode = input("Mode [1]: ").strip() or "1"
    while True:
        try:
            if mode == "2":
                txt = input("W, S, E, N [North Campus]: ").strip()
                return bbox_to_center_size(parse_bbox(txt) if txt else ex["bbox"])
            txt = input("lat, lon [Pierpont Commons]: ").strip()
            center = parse_center(txt) if txt else ex["center"]
            size_txt = input(f"size in meters [{ex['size']}]: ").strip()
            size = float(size_txt) if size_txt else ex["size"]
            return center, size
        except ValueError as exc:
            print(f"  -> {exc}. Try again.")


def resolve_area(args):
    """Priority: --bbox > --center > --preset > interactive prompt > North Campus default."""
    if args.bbox:
        return bbox_to_center_size(parse_bbox(args.bbox))
    if args.center:
        return parse_center(args.center), args.size or PRESETS["north_campus"]["size"]
    if args.preset:
        p = PRESETS[args.preset]
        return p["center"], args.size or p["size"]
    if sys.stdin.isatty() and not args.no_prompt:
        return prompt_for_area()
    p = PRESETS["north_campus"]
    return p["center"], args.size or p["size"]


# ----------------------------------------------------------------------------
# Area of interest
# ----------------------------------------------------------------------------
def make_aoi(lat, lon, size_m):
    """Square AOI of edge size_m centered on (lat, lon). Returns UTM center and lon/lat bbox."""
    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    to_ll = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    h = size_m / 2
    xs = [cx - h, cx + h, cx + h, cx - h]
    ys = [cy - h, cy - h, cy + h, cy + h]
    lons, lats = to_ll.transform(xs, ys)
    bbox = (min(lons), min(lats), max(lons), max(lats))  # west, south, east, north
    return cx, cy, bbox, to_utm, to_ll


# ----------------------------------------------------------------------------
# Terrain sources -> a sampler(lon, lat) -> elevation (m)
# ----------------------------------------------------------------------------
def _lonlat_to_global_px(lon, lat, z, tile=256):
    n = 2 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * n * tile
    lat_r = np.radians(np.asarray(lat))
    y = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n * tile
    return x, y


def terrarium_sampler(bbox, zoom):
    west, south, east, north = bbox
    gx0, gy0 = _lonlat_to_global_px(west, north, zoom)
    gx1, gy1 = _lonlat_to_global_px(east, south, zoom)
    # one-pixel margin so bilinear sampling at the edge stays inside the mosaic
    tx0, ty0 = int((gx0 - 1) // 256), int((gy0 - 1) // 256)
    tx1, ty1 = int((gx1 + 1) // 256), int((gy1 + 1) // 256)

    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), dtype=np.float32)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"[terrain] downloading {n_tiles} Terrarium tile(s) at z{zoom}")
    session = requests.Session()
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = TERRARIUM_URL.format(z=zoom, x=tx, y=ty)
            r = session.get(url, timeout=30)
            r.raise_for_status()
            rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
            elev = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
            r0, c0 = (ty - ty0) * 256, (tx - tx0) * 256
            mosaic[r0:r0 + 256, c0:c0 + 256] = elev

    def sample(lon, lat):
        gx, gy = _lonlat_to_global_px(lon, lat, zoom)
        cols = gx - tx0 * 256 - 0.5   # pixel centers sit at +0.5
        rows = gy - ty0 * 256 - 0.5
        return map_coordinates(mosaic, [rows, cols], order=1, mode="nearest")

    return sample


def dep3_sampler(bbox, res_m):
    import py3dep          # optional dependency
    import xarray as xr

    west, south, east, north = bbox
    pad_x, pad_y = (east - west) * 0.05, (north - south) * 0.05
    padded = (west - pad_x, south - pad_y, east + pad_x, north + pad_y)
    print(f"[terrain] requesting USGS 3DEP at {res_m} m")
    dem = py3dep.get_dem(padded, resolution=res_m, crs=4326)
    dem = dem.rio.reproject("EPSG:4326")

    def sample(lon, lat):
        vals = dem.interp(
            x=xr.DataArray(np.ravel(lon), dims="p"),
            y=xr.DataArray(np.ravel(lat), dims="p"),
            method="linear",
        ).values.astype(np.float64)
        if np.isnan(vals).any():
            vals[np.isnan(vals)] = np.nanmedian(vals)
        return vals

    return sample


# ----------------------------------------------------------------------------
# NAIP ground texture -> north-up RGB image covering the AOI square exactly
# ----------------------------------------------------------------------------
def _grid(size_m, res_m, max_px):
    """Pixel count and exact pixel size so the image spans the AOI edge to edge."""
    npx = int(math.ceil(size_m / res_m))
    if npx > max_px:
        print(f"[naip] {npx}px exceeds --naip-max-px {max_px}; coarsening to fit")
        npx = max_px
    return npx, size_m / npx


def fetch_naip_usgs(cx, cy, size_m, res_m, crs, year=None, max_px=8192):
    """
    Request NAIP natural color from The National Map ImageServer in the scene's
    UTM CRS. Pixel (0, 0) is the NW corner of the AOI. Tiled at 4000 px.
    """
    epsg = int(crs.split(":")[1])
    npx, res = _grid(size_m, res_m, max_px)
    h = size_m / 2
    params_base = {
        "bboxSR": epsg,
        "imageSR": epsg,
        "format": "jpgpng",
        "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "renderingRule": json.dumps({"rasterFunction": "NaturalColor"}),
        "f": "image",
    }
    if year:
        params_base["mosaicRule"] = json.dumps({
            "mosaicMethod": "esriMosaicAttribute",
            "sortField": "Year", "sortValue": str(year), "ascending": True,
            "where": f"Year = {int(year)}",
        })

    img = Image.new("RGB", (npx, npx))
    n_req = math.ceil(npx / NAIP_MAX_PX) ** 2
    print(f"[naip] USGS NAIP {npx}x{npx}px at {res:.2f} m/px "
          f"({'year ' + str(year) if year else 'newest year'}), {n_req} request(s)")
    session = requests.Session()
    for r0 in range(0, npx, NAIP_MAX_PX):
        for c0 in range(0, npx, NAIP_MAX_PX):
            w, hh = min(NAIP_MAX_PX, npx - c0), min(NAIP_MAX_PX, npx - r0)
            xmin = cx - h + c0 * res
            ymax = cy + h - r0 * res
            params = dict(params_base,
                          bbox=f"{xmin},{ymax - hh * res},{xmin + w * res},{ymax}",
                          size=f"{w},{hh}")
            r = session.get(NAIP_URL, params=params, timeout=180)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image"):
                raise RuntimeError(f"NAIP service returned non-image: {r.text[:300]}")
            tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            if tile.size != (w, hh):
                tile = tile.resize((w, hh), Image.BILINEAR)
            img.paste(tile, (c0, r0))
    return img, res


def load_naip_files(paths, cx, cy, size_m, res_m, crs, max_px=8192):
    """Reproject local NAIP GeoTIFF/JP2 tiles (e.g. EarthExplorer DOQQs) onto the AOI grid."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    npx, res = _grid(size_m, res_m, max_px)
    h = size_m / 2
    dst_transform = from_origin(cx - h, cy + h, res, res)
    out = np.zeros((3, npx, npx), dtype=np.uint8)
    for p in paths:
        tmp = np.zeros_like(out)
        with rasterio.open(p) as src:
            print(f"[naip] {Path(p).name}: {src.count} bands, {src.crs}, {src.res[0]:.2f} m")
            for b in range(3):                       # NAIP band order: R, G, B, NIR
                reproject(source=rasterio.band(src, b + 1), destination=tmp[b],
                          dst_transform=dst_transform, dst_crs=crs,
                          dst_nodata=0, resampling=Resampling.bilinear)
        has = tmp.any(axis=0)
        out[:, has] = tmp[:, has]                    # later files fill gaps / overwrite
    if not out.any(axis=0).all():
        print("[naip] warning: part of the AOI is not covered by the given files (black)")
    return Image.fromarray(np.moveaxis(out, 0, -1)), res


def save_texture(img, res, cx, cy, size_m, out):
    """JPEG + world file so the texture can be checked in QGIS against OSM."""
    h = size_m / 2
    img.save(out / "naip_texture.jpg", quality=92)
    (out / "naip_texture.jgw").write_text(
        f"{res}\n0\n0\n{-res}\n{cx - h + res / 2}\n{cy + h - res / 2}\n")


def planar_uv(vertices, size_m):
    """Top-down projection: local x,y in [-h, h] -> u,v in [0, 1] (v=1 is north)."""
    h = size_m / 2
    return np.clip((vertices[:, :2] + h) / size_m, 0.0, 1.0)


def textured(mesh, image, size_m, pbr=True):
    """Copy of mesh with NAIP draped from above. PBR for glTF, Simple for OBJ/MTL."""
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    m = mesh.copy()
    if pbr:
        mat = PBRMaterial(name="naip", baseColorTexture=image,
                          metallicFactor=0.0, roughnessFactor=1.0)
    else:
        mat = SimpleMaterial(name="naip", image=image,
                             diffuse=[255, 255, 255, 255], ambient=[255, 255, 255, 255])
    m.visual = trimesh.visual.TextureVisuals(uv=planar_uv(m.vertices, size_m), material=mat)
    return m


def matte(mesh, rgba=BUILDING_COLOR):
    """Copy with an explicit non-metallic glTF material. Without one, glTF viewers fall
    back to the spec default (metallic = 1), which renders vertex-colored walls near black."""
    from trimesh.visual.material import PBRMaterial
    m = mesh.copy()
    m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        name="building", baseColorFactor=list(rgba), metallicFactor=0.0, roughnessFactor=0.9))
    return m


def split_roofs(mesh, min_nz=0.9):
    """Split an extruded-building mesh into up-facing roof faces and everything else."""
    up = mesh.face_normals[:, 2] > min_nz
    roofs = mesh.submesh([np.nonzero(up)[0]], append=True) if up.any() else None
    walls = mesh.submesh([np.nonzero(~up)[0]], append=True) if (~up).any() else None
    if walls is not None:
        walls.unmerge_vertices()               # flat shading: no normals averaged across
        walls.visual.face_colors = BUILDING_COLOR   # corners or with the downward base
    return roofs, walls


# ----------------------------------------------------------------------------
# Roof alignment: where does each OSM roof actually appear in the NAIP image?
# ----------------------------------------------------------------------------
# An offset (dx, dy) in meters (east, north) means: the roof of the footprint at
# (x, y) is seen in the image at (x + dx, y + dy). Geometry stays on OSM; the roof
# UVs are taken at the shifted position. Pixel grid = the NAIP texture grid:
# row 0 = north edge, col 0 = west edge, pixel size `res`, local coords centered on AOI.

def _rasterize(polys, west, north, res, width, height):
    """Fill polygons (local meters) into a width x height float mask whose NW corner is (west, north)."""
    from PIL import ImageDraw
    img = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(img)

    def px(coords):  # PIL puts integer coordinates on pixel centers
        return [((x - west) / res - 0.5, (north - y) / res - 0.5) for x, y in coords]

    for p in polys:
        d.polygon(px(p.exterior.coords), fill=1)
        for ring in p.interiors:
            d.polygon(px(ring.coords), fill=0)
    return np.asarray(img, dtype=np.float32)


def _crop_box(poly, size_m, res, margin_px=2):
    """Pixel window (r0, c0, height, width) covering a polygon on the NAIP grid."""
    h = size_m / 2
    minx, miny, maxx, maxy = poly.bounds
    c0 = int(math.floor((minx + h) / res)) - margin_px
    r0 = int(math.floor((h - maxy) / res)) - margin_px
    c1 = int(math.ceil((maxx + h) / res)) + margin_px
    r1 = int(math.ceil((h - miny) / res)) + margin_px
    return r0, c0, r1 - r0, c1 - c0


def _union_mask(polys, size_m, res, npx):
    """Union of many polygons on the full grid, rasterized one crop at a time."""
    h = size_m / 2
    out = np.zeros((npx, npx), dtype=bool)
    for p in polys:
        r0, c0, hh, ww = _crop_box(p, size_m, res)
        m = _rasterize([p], -h + c0 * res, h - r0 * res, res, ww, hh) > 0
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + hh, npx), min(c0 + ww, npx)
        if re_ > rs and ce > cs:
            out[rs:re_, cs:ce] |= m[rs - r0:re_ - r0, cs - c0:ce - c0]
    return out


def _image_edges(image):
    """|d/dx| and |d/dy| of smoothed NAIP luminance, scaled to ~0..1, mean removed."""
    g = gaussian_filter(np.asarray(image.convert("L"), dtype=np.float32), 1.0)
    ex, ey = np.abs(sobel(g, axis=1)), np.abs(sobel(g, axis=0))
    s = float(np.percentile(np.hypot(ex, ey), 99)) + 1e-6
    ex, ey = np.clip(ex / s, 0, 1), np.clip(ey / s, 0, 1)
    return ex - ex.mean(), ey - ey.mean()


def _outline_edges(mask):
    """x- and y-oriented outline of a binary mask (2 px wide, 0..1)."""
    return np.abs(sobel(mask, axis=1)) / 4.0, np.abs(sobel(mask, axis=0)) / 4.0


def _parabolic_peak(W):
    """Integer argmax of a 2-D score window plus sub-pixel refinement; flags edge peaks."""
    i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    on_edge = i in (0, W.shape[0] - 1) or j in (0, W.shape[1] - 1)

    def vertex(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den < 0 else 0.0

    di = 0.0 if i in (0, W.shape[0] - 1) else vertex(W[i - 1, j], W[i, j], W[i + 1, j])
    dj = 0.0 if j in (0, W.shape[1] - 1) else vertex(W[i, j - 1], W[i, j], W[i, j + 1])
    return i + di, j + dj, float(W[i, j]), on_edge


def detect_global_offset(ex, ey, mx, my, res, search_m):
    """
    Edge cross-correlation over all footprints at once (FFT), shifts up to ±search_m.
    Score(d) = mean over footprint outline of image edge strength at outline + d,
    with x-edges matched to x-edges and y-edges to y-edges. Returns pixel shift
    (drow, dcol), peak score, score at zero shift, and whether the peak hit the window edge.
    """
    R = max(1, int(math.ceil(search_m / res)))
    H, W = ex.shape
    shape = (H + R, W + R)                       # zero padding: no wrap-around within ±R
    F = np.fft.rfft2
    corr = np.fft.irfft2(F(ex, shape) * np.conj(F(mx, shape)) +
                         F(ey, shape) * np.conj(F(my, shape)), shape)
    win = np.roll(np.roll(corr, R, 0), R, 1)[:2 * R + 1, :2 * R + 1]
    win = win / max(float(mx.sum() + my.sum()), 1e-6)
    i, j, peak, on_edge = _parabolic_peak(win)
    return (i - R, j - R), peak, float(win[R, R]), on_edge


def refine_local(records, ex, ey, size_m, res, g_px, search_m,
                 min_edge_px=150, min_gain=0.05):
    """
    Per-building search within ±search_m around the global shift. A building keeps its
    own offset only if (1) its outline is long enough to measure, (2) the best shift is
    inside the window, and (3) it beats the global shift by at least `min_gain`
    (same score units as the global search). Otherwise it keeps the global offset.
    """
    r = max(1, int(round(search_m / res)))
    gi, gj = int(round(g_px[0])), int(round(g_px[1]))
    pad = r + max(abs(gi), abs(gj)) + 4
    exp, eyp = np.pad(ex, pad), np.pad(ey, pad)
    h = size_m / 2
    n_local = 0
    for rec in records:
        r0, c0, hh, ww = _crop_box(rec["poly"], size_m, res)
        m = _rasterize([rec["poly"]], -h + c0 * res, h - r0 * res, res, ww, hh)
        mx, my = _outline_edges(m)
        rr, cc = np.nonzero((mx + my) > 0)
        if len(rr) < min_edge_px:
            continue
        wx, wy = mx[rr, cc], my[rr, cc]
        norm = float(wx.sum() + wy.sum())
        rr, cc = rr + r0 + pad + gi, cc + c0 + pad + gj
        rr, cc = np.clip(rr, r, exp.shape[0] - r - 1), np.clip(cc, r, exp.shape[1] - r - 1)
        W = np.empty((2 * r + 1, 2 * r + 1))
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                W[a + r, b + r] = (wx @ exp[rr + a, cc + b] + wy @ eyp[rr + a, cc + b]) / norm
        i, j, best, on_edge = _parabolic_peak(W)
        gain = best - float(W[r, r])
        rec["roof_gain"] = round(gain, 4)
        if on_edge or gain < min_gain:
            continue
        drow, dcol = gi + i - r, gj + j - r
        rec["roof_dx"], rec["roof_dy"] = dcol * res, -drow * res
        rec["roof_align"] = "local"
        n_local += 1
    return n_local


def align_roofs(records, image, res, size_m, mode="global", manual=None,
                search_m=15.0, local_search_m=3.0):
    """Set roof_dx / roof_dy / roof_align on every record. Returns the global (dx, dy) in m."""
    for rec in records:
        rec.update(roof_dx=0.0, roof_dy=0.0, roof_align="none")
    if mode == "off" or not records:
        return 0.0, 0.0
    if manual is not None:
        dx, dy = manual
        for rec in records:
            rec.update(roof_dx=dx, roof_dy=dy, roof_align="manual")
        print(f"[roofs] manual offset dx={dx:+.2f} m east, dy={dy:+.2f} m north")
        return dx, dy

    npx = image.size[0]
    ex, ey = _image_edges(image)
    mx, my = _outline_edges(_union_mask([r["poly"] for r in records], size_m, res, npx)
                            .astype(np.float32))
    (drow, dcol), peak, zero, on_edge = detect_global_offset(ex, ey, mx, my, res, search_m)
    dx, dy = dcol * res, -drow * res
    print(f"[roofs] global offset dx={dx:+.2f} m east, dy={dy:+.2f} m north "
          f"(edge match {peak:.3f} vs {zero:.3f} unshifted)")
    if on_edge:
        print(f"[roofs] WARNING: best match is at the ±{search_m:g} m search limit, so it is "
              f"probably spurious. Using no offset; check roof_alignment_preview.jpg and "
              f"set --roof-offset by hand, or raise --roof-search.")
        dx = dy = 0.0
        drow = dcol = 0.0
    elif peak - zero < 0.01:
        print("[roofs] note: the shift barely improves the match; footprints may already "
              "line up, or the image has too little contrast at roof edges.")
    for rec in records:
        rec.update(roof_dx=dx, roof_dy=dy, roof_align="global")

    if mode == "local":
        n = refine_local(records, ex, ey, size_m, res, (drow, dcol), local_search_m)
        print(f"[roofs] per-building refinement: {n} of {len(records)} footprints use their "
              f"own offset (±{local_search_m:g} m around global); the rest keep the global one")
    return dx, dy


def textured_roofs(records, image, size_m, pbr=True, min_nz=0.9):
    """Up-facing faces of every building, UV-mapped at that building's roof offset."""
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    V, F, UV, n = [], [], [], 0
    for rec in records:
        m = rec.get("mesh")
        if m is None:
            continue
        up = np.nonzero(m.face_normals[:, 2] > min_nz)[0]
        if not len(up):
            continue
        sub = m.submesh([up], append=True)
        shift = np.array([rec.get("roof_dx", 0.0), rec.get("roof_dy", 0.0), 0.0])
        V.append(sub.vertices)
        F.append(sub.faces + n)
        UV.append(planar_uv(sub.vertices + shift, size_m))
        n += len(sub.vertices)
    if not V:
        return None
    roofs = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    if pbr:
        mat = PBRMaterial(name="naip_roofs", baseColorTexture=image,
                          metallicFactor=0.0, roughnessFactor=1.0)
    else:
        mat = SimpleMaterial(name="naip_roofs", image=image,
                             diffuse=[255, 255, 255, 255], ambient=[255, 255, 255, 255])
    roofs.visual = trimesh.visual.TextureVisuals(uv=np.vstack(UV), material=mat)
    return roofs


def inpaint_roofs(image, records, size_m, res, grow_m=1.2):
    """
    Ground drape with roofs removed: fill both the footprint and the place the roof
    appears in the image (plus a small margin for eaves / walls / shadow edges),
    so no ghost roof shows on the ground beside a building.
    """
    npx = image.size[0]
    polys = []
    for rec in records:
        polys.append(rec["poly"])
        if rec.get("roof_dx", 0.0) or rec.get("roof_dy", 0.0):
            polys.append(shp_translate(rec["poly"], xoff=rec["roof_dx"], yoff=rec["roof_dy"]))
    mask = _union_mask(polys, size_m, res, npx)
    k = max(1, int(round(grow_m / res)))
    mask = binary_dilation(mask, iterations=k)
    rgb = np.asarray(image.convert("RGB"))
    try:
        import cv2
        out = cv2.inpaint(np.ascontiguousarray(rgb), mask.astype(np.uint8), 8, cv2.INPAINT_TELEA)
    except ImportError:                       # no OpenCV: nearest-ground fill, then soften
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        filled = rgb[idx[0], idx[1]].astype(np.float32)
        soft = np.stack([gaussian_filter(filled[..., b], 3.0) for b in range(3)], axis=-1)
        out = np.where(mask[..., None], soft, rgb).astype(np.uint8)
    print(f"[roofs] ground drape: inpainted {mask.mean() * 100:.1f}% of the image under/around roofs")
    return Image.fromarray(out)


def save_roof_preview(image, records, size_m, res, path):
    """NAIP with outlines: red = OSM footprint, cyan = global offset, yellow = per-building."""
    from PIL import ImageDraw
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    h = size_m / 2
    width = max(1, round(im.size[0] / 2500))
    colors = {"global": (0, 255, 255), "manual": (0, 255, 255), "local": (255, 230, 0)}
    for rec in records:
        shifted = shp_translate(rec["poly"], xoff=rec.get("roof_dx", 0.0), yoff=rec.get("roof_dy", 0.0))
        for poly, col in ((rec["poly"], (255, 40, 40)),
                          (shifted, colors.get(rec.get("roof_align"), (255, 40, 40)))):
            for ring in [poly.exterior, *poly.interiors]:
                d.line([((x + h) / res - 0.5, (h - y) / res - 0.5) for x, y in ring.coords],
                       fill=col, width=width)
    im.save(path, quality=90)


# ----------------------------------------------------------------------------
# Terrain mesh
# ----------------------------------------------------------------------------
def build_terrain(sampler, cx, cy, size_m, spacing, to_ll, exaggeration):
    h = size_m / 2
    xs = np.arange(-h, h + 1e-6, spacing)
    ys = np.arange(-h, h + 1e-6, spacing)
    XX, YY = np.meshgrid(xs, ys)                      # shape (ny, nx); row = y, col = x
    lon, lat = to_ll.transform(XX.ravel() + cx, YY.ravel() + cy)
    Z = np.asarray(sampler(lon, lat), dtype=np.float64).reshape(XX.shape)

    z_datum = float(Z.min())                          # local vertical origin
    Zrel = (Z - z_datum) * exaggeration
    print(f"[terrain] grid {len(xs)}x{len(ys)}, elevation {Z.min():.1f}–{Z.max():.1f} m "
          f"(relief {Z.max() - Z.min():.1f} m)")

    ny, nx = XX.shape
    verts = np.column_stack([XX.ravel(), YY.ravel(), Zrel.ravel()])
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])  # +Z normals

    interp = RegularGridInterpolator((ys, xs), Zrel, bounds_error=False, fill_value=None)
    return verts, faces, nx, ny, interp, z_datum


def make_solid(verts, faces, nx, ny, base_z):
    """Add side walls and a bottom so the terrain is a closed, printable slab."""
    n = len(verts)
    bottom = verts.copy()
    bottom[:, 2] = base_z
    V = np.vstack([verts, bottom])

    idx = np.arange(ny * nx).reshape(ny, nx)
    loop = np.concatenate([            # boundary, counter-clockwise seen from above
        idx[0, :],
        idx[1:, -1],
        idx[-1, -2::-1],
        idx[-2:0:-1, 0],
    ])
    p, q = loop, np.roll(loop, -1)
    walls = np.vstack([
        np.column_stack([p + n, q + n, q]),
        np.column_stack([p + n, q, p]),
    ])
    bottom_faces = faces[:, ::-1] + n  # reversed winding -> -Z normals
    return V, np.vstack([faces, walls, bottom_faces])


def color_terrain(mesh):
    z = mesh.triangles_center[:, 2]
    t = (z - z.min()) / max(np.ptp(z), 1e-6)
    rgb = TERRAIN_COLOR_LOW + (TERRAIN_COLOR_HIGH - TERRAIN_COLOR_LOW) * t[:, None]
    rgba = np.column_stack([rgb, np.full(len(rgb), 255)]).astype(np.uint8)
    mesh.visual.face_colors = rgba


# ----------------------------------------------------------------------------
# OSM buildings
# ----------------------------------------------------------------------------
def parse_length(value):
    """Parse OSM length tags like '12', '12 m', '40 ft', "40'"."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip().lower()
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    if "ft" in s or "'" in s:
        v *= 0.3048
    return v


def building_heights(row):
    """Return (min_height, height) in meters above the footprint's lowest ground point."""
    get = lambda k: row[k] if k in row.index else None
    height = parse_length(get("height"))
    if height is None:
        levels = parse_length(get("building:levels"))
        roof_levels = parse_length(get("roof:levels")) or 0.0
        if levels is not None:
            height = (levels + roof_levels) * LEVEL_HEIGHT_M
    if height is None or height <= 0:
        height = DEFAULT_HEIGHT_M

    min_h = parse_length(get("min_height"))
    if min_h is None:
        min_level = parse_length(get("building:min_level"))
        min_h = min_level * LEVEL_HEIGHT_M if min_level is not None else 0.0
    min_h = min(min_h, height - 0.5)
    return max(min_h, 0.0), height


def height_source(row):
    """Which OSM tag the height came from: 'height', 'levels' or 'default'."""
    get = lambda k: row[k] if k in row.index else None
    if parse_length(get("height")) is not None:
        return "height"
    if parse_length(get("building:levels")) is not None:
        return "levels"
    return "default"


def fetch_buildings(bbox):
    import osmnx as ox
    ox.settings.use_cache = True
    west, south, east, north = bbox
    tags = {"building": True}
    print("[osm] querying Overpass for buildings")
    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)  # osmnx >= 2.0
    except TypeError:
        gdf = ox.features_from_bbox(north, south, east, west, tags=tags)          # osmnx 1.x
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    print(f"[osm] {len(gdf)} building polygons")
    return gdf.to_crs(UTM_CRS)


def footprint_samples(poly, step):
    """Points on the outline (densified) and on a grid inside the footprint."""
    import shapely
    ring = np.asarray(poly.exterior.segmentize(step).coords)[:, :2]
    minx, miny, maxx, maxy = poly.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if len(pts):
        pts = pts[shapely.contains_xy(poly, pts[:, 0], pts[:, 1])]
    return np.vstack([ring, pts])


def build_buildings(gdf, cx, cy, size_m, terrain_interp, sample_step, ground_pct):
    """
    Platform + foundation placement:
      roof      = ground_high + height   (ground_high = highest terrain under the footprint)
      bottom    = ground_low - EMBED_M   (foundation fills the downhill gap)
    Raised parts (min_height > 0) start at ground_high + min_height with no foundation.
    """
    clip = box(-size_m / 2, -size_m / 2, size_m / 2, size_m / 2)
    meshes, records, skipped = [], [], 0

    for idx, row in gdf.iterrows():
        geom = shp_translate(row.geometry, xoff=-cx, yoff=-cy).intersection(clip)
        if geom.is_empty:
            continue
        polys = [geom] if isinstance(geom, Polygon) else (
            list(geom.geoms) if isinstance(geom, MultiPolygon) else
            [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
        )
        min_h, height = building_heights(row)
        src = height_source(row)
        osm_id = "/".join(str(v) for v in idx) if isinstance(idx, tuple) else str(idx)
        name = row["name"] if "name" in row.index and isinstance(row["name"], str) else ""

        for poly in polys:
            if poly.area < 4.0:        # ignore slivers from clipping
                continue
            try:
                pts = footprint_samples(poly, sample_step)
                ground = terrain_interp(pts[:, ::-1])        # interpolator expects (y, x)
                ground_low = float(np.min(ground))
                ground_high = float(np.percentile(ground, ground_pct))

                top = ground_high + height
                if min_h > 0:
                    bottom = ground_high + min_h             # raised part: leave space below
                else:
                    bottom = ground_low - EMBED_M            # building + foundation, one prism

                m = trimesh.creation.extrude_polygon(poly, height=top - bottom)
                m.apply_translation([0, 0, bottom])
                meshes.append(m)
                records.append({"poly": poly, "osm_id": osm_id, "name": name,
                                "height": height, "min_height": min_h,
                                "height_source": src, "base_z": bottom, "top_z": top,
                                "mesh": m})
            except Exception as exc:   # noqa: BLE001 - keep going on bad footprints
                skipped += 1
                if skipped <= 5:
                    print(f"[osm] skipped a footprint: {exc}")

    print(f"[osm] extruded {len(meshes)} building meshes ({skipped} skipped)")
    if records:
        counts = {k: sum(r["height_source"] == k for r in records)
                  for k in ("height", "levels", "default")}
        print(f"[osm] height from: height tag {counts['height']}, levels tag {counts['levels']}, "
              f"{DEFAULT_HEIGHT_M:g} m default {counts['default']}")
    if not meshes:
        return None, records
    merged = trimesh.util.concatenate(meshes)
    merged.visual.face_colors = BUILDING_COLOR
    return merged, records


def save_footprints(records, cx, cy, z_datum, crs, path):
    """GeoJSON of the footprints actually used, in UTM meters (for check_naip_offsets.py / QGIS)."""
    from shapely.geometry import mapping
    feats = []
    for r in records:
        g = shp_translate(r["poly"], xoff=cx, yoff=cy)
        props = {k: r[k] for k in ("osm_id", "name", "height", "min_height", "height_source")}
        props["base_elev_m"] = round(r["base_z"] + z_datum, 2)
        props["roof_elev_m"] = round(r["top_z"] + z_datum, 2)
        if "roof_align" in r:          # where the roof texture was sampled (m east / north)
            props["roof_dx_m"] = round(r["roof_dx"], 2)
            props["roof_dy_m"] = round(r["roof_dy"], 2)
            props["roof_align"] = r["roof_align"]
            if "roof_gain" in r:
                props["roof_local_gain"] = r["roof_gain"]
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": props})
    epsg = crs.split(":")[1]
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
        "features": feats}))


# ----------------------------------------------------------------------------
# Terrain vs. image: hillshade on the same pixel grid as the NAIP texture
# ----------------------------------------------------------------------------
def hillshade(Z, spacing, azimuth=315.0, altitude=45.0):
    """Z rows run south->north (as built). Returns a north-up uint8 hillshade."""
    dzdy, dzdx = np.gradient(Z, spacing)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, -dzdy)          # direction the slope faces, from north
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return (np.clip(hs, 0, 1) * 255).astype(np.uint8)[::-1]


def save_hillshade(Zgrid, spacing, naip, out):
    hs = Image.fromarray(hillshade(Zgrid, spacing))
    size = naip.size if naip is not None else hs.size
    hs = hs.resize(size, Image.BILINEAR)
    hs.save(out / "terrain_hillshade.png")
    if naip is not None:
        Image.blend(naip.convert("RGB"), hs.convert("RGB"), 0.5).save(
            out / "naip_hillshade_blend.jpg", quality=90)
    return hs


# ----------------------------------------------------------------------------
# Google Earth export: KMZ with COLLADA models in true east-north-up
# ----------------------------------------------------------------------------
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def local_to_enu(xyz, cx, cy, z_datum, to_ll, lon0, lat0):
    """
    Local model coords (UTM offsets from AOI center, z above z_datum) -> east/north/up
    meters about (lon0, lat0, z_datum). Removes the UTM grid-north rotation (about 1.8 deg
    at Ann Arbor) and UTM scale, which would otherwise smear the model by ~10-20 m at
    the AOI edges once Google Earth places it in its local tangent frame.
    """
    lon, lat = to_ll.transform(xyz[:, 0] + cx, xyz[:, 1] + cy)
    X, Y, Z = _TO_ECEF.transform(lon, lat, xyz[:, 2] + z_datum)
    X0, Y0, Z0 = _TO_ECEF.transform(lon0, lat0, z_datum)
    d = np.column_stack([X - X0, Y - Y0, Z - Z0])
    p, l = math.radians(lat0), math.radians(lon0)
    R = np.array([
        [-math.sin(l), math.cos(l), 0.0],
        [-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)],
        [math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p)],
    ])
    return d @ R.T


def _floats(a, fmt="%.3f"):
    return " ".join(fmt % v for v in np.asarray(a).ravel())


def collada(verts, faces, normals, uv=None, image=None, color=(0.84, 0.84, 0.86)):
    """Minimal COLLADA 1.4.1, Z-up, meters. One material; texture if uv + image given."""
    nv, nf = len(verts), len(faces)
    tex = uv is not None and image is not None
    if tex:
        images = f'<library_images><image id="img"><init_from>{image}</init_from></image></library_images>'
        diffuse = '<texture texture="samp" texcoord="UVSET0"/>'
        params = ('<newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>'
                  '<newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>')
        uv_src = (f'<source id="uv"><float_array id="uv-a" count="{nv * 2}">{_floats(uv, "%.6f")}</float_array>'
                  f'<technique_common><accessor source="#uv-a" count="{nv}" stride="2">'
                  '<param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>')
        uv_in = '<input semantic="TEXCOORD" source="#uv" offset="0" set="0"/>'
        bind = '<bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/>'
    else:
        images = params = uv_src = uv_in = bind = ""
        diffuse = "<color>%.3f %.3f %.3f 1</color>" % tuple(color)
    xyz = '<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
{images}
<library_effects><effect id="fx"><profile_COMMON>{params}<technique sid="common"><lambert><diffuse>{diffuse}</diffuse></lambert></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
<library_geometries><geometry id="geom"><mesh>
<source id="pos"><float_array id="pos-a" count="{nv * 3}">{_floats(verts)}</float_array><technique_common><accessor source="#pos-a" count="{nv}" stride="3">{xyz}</accessor></technique_common></source>
<source id="nrm"><float_array id="nrm-a" count="{nv * 3}">{_floats(normals, "%.4f")}</float_array><technique_common><accessor source="#nrm-a" count="{nv}" stride="3">{xyz}</accessor></technique_common></source>
{uv_src}
<vertices id="vtx"><input semantic="POSITION" source="#pos"/><input semantic="NORMAL" source="#nrm"/></vertices>
<triangles material="m" count="{nf}"><input semantic="VERTEX" source="#vtx" offset="0"/>{uv_in}<p>{" ".join(map(str, np.asarray(faces).ravel()))}</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="model"><instance_geometry url="#geom"><bind_material><technique_common><instance_material symbol="m" target="#mat">{bind}</instance_material></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""


def _grid_surface(interp, size_m, spacing):
    """Top terrain surface resampled at `spacing` (local coords, +Z faces)."""
    h = size_m / 2
    xs = np.linspace(-h, h, int(round(size_m / spacing)) + 1)
    XX, YY = np.meshgrid(xs, xs)
    Z = interp(np.column_stack([YY.ravel(), XX.ravel()]))
    ny, nx = XX.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return np.column_stack([XX.ravel(), YY.ravel(), Z]), faces


def export_kmz(path, *, terrain_interp, buildings, naip, hillshade_img, size_m, cx, cy,
               z_datum, to_ll, lat0, lon0, spacing=5.0, altitude="absolute",
               z_offset=0.0, tex_max=4096, title="OSM buildings on terrain",
               naip_overlay=None, roofs=None, roof_image=None):
    """
    Google Earth Pro KMZ:
      Model folder   terrain (NAIP-textured) + buildings, placed with <Model> at the AOI center
      Overlays       NAIP and terrain hillshade draped on Google's own terrain (off by default)
      AOI outline and a center placemark with our ground elevation for spot checks
    altitude='absolute'  model z=0 sits at z_datum m above mean sea level (our DEM's datum)
    altitude='anchor'    our ground at the AOI center is pinned to Google's ground there
    """
    files = {}

    def jpeg(img):
        tex = img.convert("RGB")
        if max(tex.size) > tex_max:
            tex = tex.resize((tex_max, tex_max), Image.BILINEAR)
        buf = io.BytesIO(); tex.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    # terrain model
    tv_local, tf = _grid_surface(terrain_interp, size_m, spacing)
    tv = local_to_enu(tv_local, cx, cy, z_datum, to_ll, lon0, lat0)
    tmesh = trimesh.Trimesh(tv, tf, process=False)
    if naip is not None:
        files["models/naip.jpg"] = jpeg(naip)
        files["models/terrain.dae"] = collada(tv, tf, tmesh.vertex_normals,
                                              uv=planar_uv(tv_local, size_m), image="naip.jpg")
    else:
        files["models/terrain.dae"] = collada(tv, tf, tmesh.vertex_normals, color=(0.55, 0.6, 0.45))

    # buildings model: unshared vertices so each wall/roof is flat-shaded
    if buildings is not None:
        bv_local = buildings.vertices[buildings.faces].reshape(-1, 3)
        bf = np.arange(len(bv_local)).reshape(-1, 3)
        bv = local_to_enu(bv_local, cx, cy, z_datum, to_ll, lon0, lat0)
        bmesh = trimesh.Trimesh(bv, bf, process=False)
        bn = np.repeat(bmesh.face_normals, 3, axis=0)
        files["models/buildings.dae"] = collada(bv, bf, bn)

    # roofs with NAIP texture (UVs already carry each building's roof offset)
    if roofs is not None and roof_image is not None:
        rv = local_to_enu(roofs.vertices, cx, cy, z_datum, to_ll, lon0, lat0)
        rmesh = trimesh.Trimesh(rv, roofs.faces, process=False)
        files["models/naip_roofs.jpg"] = jpeg(roof_image)
        files["models/roofs.dae"] = collada(rv, roofs.faces, rmesh.vertex_normals,
                                            uv=roofs.visual.uv, image="naip_roofs.jpg")
    overlay_img = naip_overlay if naip_overlay is not None else naip
    if overlay_img is not None and overlay_img is not naip:
        files["files/naip_raw.jpg"] = jpeg(overlay_img)
    overlay_href = "files/naip_raw.jpg" if "files/naip_raw.jpg" in files else "models/naip.jpg"

    if hillshade_img is not None:
        buf = io.BytesIO(); hillshade_img.convert("L").save(buf, "PNG")
        files["files/hillshade.png"] = buf.getvalue()

    # placement
    if altitude == "anchor":
        z_center = float(terrain_interp([[0.0, 0.0]])[0])
        alt_mode, alt = "relativeToGround", -z_center + z_offset
    else:
        alt_mode, alt = "absolute", z_datum + z_offset

    h = size_m / 2
    cl, ca = to_ll.transform([cx - h, cx + h, cx + h, cx - h], [cy - h, cy - h, cy + h, cy + h])
    quad = " ".join(f"{lo:.8f},{la:.8f}" for lo, la in zip(cl, ca))   # LL, LR, UR, UL
    ring = quad + f" {cl[0]:.8f},{ca[0]:.8f}"
    z_c = float(terrain_interp([[0.0, 0.0]])[0]) + z_datum

    def model(name, href, tex=None):
        rmap = (f'<ResourceMap><Alias><targetHref>models/{tex}</targetHref>'
                f'<sourceHref>{tex}</sourceHref></Alias></ResourceMap>') if tex else ""
        return f"""<Placemark><name>{escape(name)}</name><Model>
  <altitudeMode>{alt_mode}</altitudeMode>
  <Location><longitude>{lon0:.8f}</longitude><latitude>{lat0:.8f}</latitude><altitude>{alt:.3f}</altitude></Location>
  <Orientation><heading>0</heading><tilt>0</tilt><roll>0</roll></Orientcat osmation>
  <Scale><x>1</x><y>1</y><z>1</z></Scale>
  <Link><href>{href}</href></Link>{rmap}
</Model></Placemark>"""

    def overlay(name, href, visible, desc):
        return f"""<GroundOverlay><name>{escape(name)}</name><visibility>{visible}</visibility>
  <description>{escape(desc)}</description><color>ffffffff</color>
  <Icon><href>{href}</href></Icon><gx:LatLonQuad><coordinates>{quad}</coordinates></gx:LatLonQuad>
</GroundOverlay>"""

    parts = [model("Terrain" + (" (NAIP texture)" if naip is not None else ""),
                   "models/terrain.dae", "naip.jpg" if naip is not None else None)]
    if buildings is not None:
        parts.append(model("OSM buildings" + (" (walls)" if "models/roofs.dae" in files else ""),
                           "models/buildings.dae"))
    if "models/roofs.dae" in files:
        parts.append(model("Roofs (NAIP texture)", "models/roofs.dae", "naip_roofs.jpg"))
    overlays = []
    if overlay_img is not None:
        overlays.append(overlay("NAIP on Google terrain", overlay_href, 0,
                                "Same NAIP image draped on Google's terrain. Toggle it against "
                                "Google imagery to see horizontal offsets."))
    if hillshade_img is not None:
        overlays.append(overlay("Our terrain hillshade", "files/hillshade.png", 0,
                                "Hillshade of the model's terrain, draped on Google's terrain."))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document><name>{escape(title)}</name><open>1</open>
<description>{escape(f"Model z=0 = {z_datum:.2f} m; altitude mode {altitude}; terrain resampled at {spacing:g} m.")}</description>
<Style id="aoi"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
<LookAt><longitude>{lon0:.8f}</longitude><latitude>{lat0:.8f}</latitude><altitude>0</altitude>
  <heading>20</heading><tilt>55</tilt><range>{size_m * 1.4:.0f}</range><altitudeMode>relativeToGround</altitudeMode></LookAt>
<Folder><name>Model</name><open>1</open>
{chr(10).join(parts)}
</Folder>
<Folder><name>Overlays (on Google terrain)</name>
{chr(10).join(overlays)}
</Folder>
<Placemark><name>AOI outline</name><styleUrl>#aoi</styleUrl><Polygon><tessellate>1</tessellate>
  <altitudeMode>clampToGround</altitudeMode><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
<Placemark><name>Center: our ground {z_c:.1f} m</name>
  <description>{escape(f"Our terrain elevation here: {z_c:.2f} m. Compare with the elevation Google Earth shows in its status bar at this point.")}</description>
  <Point><coordinates>{lon0:.8f},{lat0:.8f},0</coordinates></Point></Placemark>
</Document></kml>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)                    # must be the first entry
        for name, data in files.items():
            z.writestr(name, data)
    ntri = (len(tf) + (len(buildings.faces) if buildings is not None else 0)
            + (len(roofs.faces) if roofs is not None else 0))
    print(f"[kmz] {path.name}: {ntri:,} triangles, terrain at {spacing:g} m, altitude {altitude}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def export_textured_obj(mesh, path):
    """OBJ + MTL + texture image side by side."""
    from trimesh.exchange.obj import export_obj
    mtl = path.with_suffix(".mtl").name
    text, files = export_obj(mesh, include_texture=True, return_texture=True, mtl_name=mtl)
    path.write_text(text)
    for name, data in files.items():
        (path.parent / name).write_bytes(data)


def main():
    ap = argparse.ArgumentParser(description="OSM 3D buildings on NAIP-textured terrain")
    ap.add_argument("--center", help='center point "lat, lon", e.g. "42.2917, -83.7174"')
    ap.add_argument("--bbox", help='"west, south, east, north", e.g. "-83.7250, 42.2860, -83.7090, 42.2990"')
    ap.add_argument("--preset", choices=sorted(PRESETS), help="use an example area")
    ap.add_argument("--no-prompt", action="store_true", help="never ask; fall back to North Campus")
    ap.add_argument("--size", type=float, default=None, help="AOI edge length, m (with --center/--preset)")
    ap.add_argument("--spacing", type=float, default=2.0, help="terrain grid spacing, m")
    ap.add_argument("--source", choices=["terrarium", "3dep"], default="terrarium")
    ap.add_argument("--zoom", type=int, default=15, help="Terrarium zoom (max 15)")
    ap.add_argument("--dem-res", type=int, default=1, help="3DEP resolution, m")
    ap.add_argument("--exaggeration", type=float, default=1.0, help="terrain vertical scale")
    ap.add_argument("--solid", action="store_true", help="close terrain into a printable slab")
    ap.add_argument("--base", type=float, default=5.0, help="slab thickness below lowest point, m")
    ap.add_argument("--ground-pct", type=float, default=100,
                    help="percentile of footprint ground used as the building platform "
                         "(100 = highest point; try 90 to ignore DEM spikes)")
    # NAIP texture
    ap.add_argument("--no-naip", action="store_true", help="skip NAIP; use height colors")
    ap.add_argument("--naip-file", nargs="+", help="local NAIP GeoTIFF/JP2 tile(s) instead of the web service")
    ap.add_argument("--naip-year", type=int, help="lock the web service to one NAIP year, e.g. 2022")
    ap.add_argument("--naip-res", type=float, default=0.6, help="texture pixel size, m (NAIP native 0.6)")
    ap.add_argument("--naip-max-px", type=int, default=8192,
                    help="cap on texture edge in pixels (GPU texture limit)")
    ap.add_argument("--roof-texture", action="store_true",
                    help="also project NAIP onto building roofs (walls stay gray)")
    ap.add_argument("--roof-align", choices=["global", "local", "off"], default="global",
                    help="roof texture offset: one for the AOI (global), global + per-building "
                         "refinement (local), or none (off)")
    ap.add_argument("--roof-offset", help='manual roof offset "dx, dy" in meters (east, north); '
                                          'skips detection')
    ap.add_argument("--roof-search", type=float, default=15.0,
                    help="max global roof offset searched, m")
    ap.add_argument("--roof-local-search", type=float, default=3.0,
                    help="per-building search radius around the global offset, m")
    ap.add_argument("--no-ground-inpaint", action="store_true",
                    help="with --roof-texture: keep roofs in the ground drape")
    # Google Earth
    ap.add_argument("--no-kmz", action="store_true", help="skip the Google Earth KMZ")
    ap.add_argument("--kmz-spacing", type=float, default=5.0,
                    help="terrain grid spacing in the KMZ, m (coarser = lighter in Google Earth)")
    ap.add_argument("--kmz-altitude", choices=["absolute", "anchor"], default="absolute",
                    help="absolute: our elevations as-is; anchor: pin our ground at the AOI "
                         "center to Google's ground")
    ap.add_argument("--kmz-offset", type=float, default=0.0,
                    help="raise (+) or lower (-) the whole model in Google Earth, m")
    ap.add_argument("--out", default=None,
                    help="output folder (default: output_roof_textures/ with --roof-texture, "
                         "output_no_roof_textures/ without)")
    args = ap.parse_args()

    if args.out is None:          # keep roof-texture runs apart from plain runs and each other
        if args.roof_texture:
            mode = "manual" if args.roof_offset else args.roof_align
            args.out = {"global": "output_roof_textures",
                        "local": "output_roof_textures_per_building",
                        "manual": "output_roof_textures_manual_shift",
                        "off": "output_roof_textures_no_shift"}[mode]
        else:
            args.out = "output_no_roof_textures"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    global UTM_CRS
    try:
        (lat, lon), size = resolve_area(args)
    except ValueError as exc:
        ap.error(str(exc))
    if size > 5000:
        print(f"[warn] {size:.0f} m is large; Overpass and the mesh may be slow. Consider --spacing 5")
    args.size = size
    UTM_CRS = utm_crs_for(lat, lon)
    print(f"[aoi] center {lat:.6f}, {lon:.6f} | size {size:.0f} m | {UTM_CRS}")

    cx, cy, bbox, _, to_ll = make_aoi(lat, lon, size)
    print(f"[aoi] bbox W,S,E,N = {tuple(round(v, 6) for v in bbox)}")

    if args.source == "3dep":
        try:
            sampler = dep3_sampler(bbox, args.dem_res)
        except Exception as exc:  # noqa: BLE001
            print(f"[terrain] 3DEP failed ({exc}); falling back to Terrarium")
            sampler = terrarium_sampler(bbox, args.zoom)
    else:
        sampler = terrarium_sampler(bbox, args.zoom)

    verts, faces, nx, ny, interp, z_datum = build_terrain(
        sampler, cx, cy, args.size, args.spacing, to_ll, args.exaggeration)
    Zgrid = verts[:, 2].reshape(ny, nx).copy()        # top surface, rows south -> north
    gx = np.unique(verts[:, 0])[:nx] + cx
    gy = np.unique(verts[:, 1])[:ny] + cy
    np.savez_compressed(out / "terrain_grid.npz", x=gx, y=gy,
                        z=Zgrid / args.exaggeration + z_datum, crs=UTM_CRS,
                        source=args.source, spacing=args.spacing)
    if args.solid:
        verts, faces = make_solid(verts, faces, nx, ny, base_z=-args.base)
    terrain = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    color_terrain(terrain)

    manual_offset = None
    if args.roof_offset:
        vals = _numbers(args.roof_offset)
        if len(vals) != 2:
            ap.error('--roof-offset expects "dx, dy" in meters, e.g. "5.4, -0.6"')
        manual_offset = tuple(vals)

    # NAIP ground texture (falls back to height colors on any failure)
    naip, res = None, None
    if not args.no_naip:
        try:
            if args.naip_file:
                naip, res = load_naip_files(args.naip_file, cx, cy, args.size,
                                            args.naip_res, UTM_CRS, args.naip_max_px)
            else:
                naip, res = fetch_naip_usgs(cx, cy, args.size, args.naip_res, UTM_CRS,
                                            args.naip_year, args.naip_max_px)
            save_texture(naip, res, cx, cy, args.size, out)
        except Exception as exc:  # noqa: BLE001
            print(f"[naip] failed ({exc}); using height-based terrain colors")
            naip = None

    hs_img = save_hillshade(Zgrid, args.spacing, naip, out)

    buildings, records = build_buildings(fetch_buildings(bbox), cx, cy, args.size, interp,
                                         sample_step=args.spacing, ground_pct=args.ground_pct)

    # Roof textures: find where each roof sits in the NAIP image, UV-map it from there
    roofs = walls = None
    ground_img = naip                                 # what gets draped on the terrain
    if args.roof_texture and naip is None:
        print("[roofs] --roof-texture needs NAIP; buildings stay gray")
    if args.roof_texture and naip is not None and buildings is not None:
        align_roofs(records, naip, res, args.size, mode=args.roof_align, manual=manual_offset,
                    search_m=args.roof_search, local_search_m=args.roof_local_search)
        save_roof_preview(naip, records, args.size, res, out / "roof_alignment_preview.jpg")
        roofs = textured_roofs(records, naip, args.size)
        _, walls = split_roofs(buildings)
        if not args.no_ground_inpaint:
            ground_img = inpaint_roofs(naip, records, args.size, res)
            ground_img.save(out / "naip_ground_inpainted.jpg", quality=92)
    save_footprints(records, cx, cy, z_datum, UTM_CRS, out / "buildings_footprints_utm.geojson")

    # Z-up single-mesh exports (meters, origin = AOI center, z=0 = lowest terrain point)
    parts = [terrain] + ([buildings] if buildings is not None else [])
    combined = trimesh.util.concatenate(parts)
    combined.export(out / "north_campus_combined.obj")
    combined.export(out / "north_campus_combined.stl")

    # Scene for viewers
    scene = trimesh.Scene()
    if ground_img is not None:
        scene.add_geometry(textured(terrain, ground_img, args.size),
                           node_name="terrain", geom_name="terrain")
        export_textured_obj(textured(terrain, ground_img, args.size, pbr=False),
                            out / "north_campus_terrain_naip.obj")
    else:
        scene.add_geometry(terrain, node_name="terrain", geom_name="terrain")

    if buildings is not None:
        if roofs is not None:
            scene.add_geometry(roofs, node_name="roofs", geom_name="roofs")
            if walls is not None:
                scene.add_geometry(matte(walls), node_name="walls", geom_name="walls")
        else:
            scene.add_geometry(matte(buildings), node_name="buildings", geom_name="buildings")

    # Y-up GLB (glTF convention): rotate -90 deg about X
    scene.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    scene.export(out / "north_campus_scene.glb")

    if not args.no_kmz:
        lon0, lat0 = to_ll.transform(cx, cy)
        export_kmz(out / "north_campus_google_earth.kmz", terrain_interp=interp,
                   buildings=walls if roofs is not None else buildings,
                   naip=ground_img, naip_overlay=naip, roofs=roofs, roof_image=naip,
                   hillshade_img=hs_img, size_m=args.size,
                   cx=cx, cy=cy, z_datum=z_datum, to_ll=to_ll, lat0=lat0, lon0=lon0,
                   spacing=args.kmz_spacing, altitude=args.kmz_altitude,
                   z_offset=args.kmz_offset)

    print(f"[done] wrote to {out.resolve()}")
    print(f"       local origin = {UTM_CRS} ({cx:.2f}, {cy:.2f}), z=0 -> {z_datum:.2f} m elevation")


if __name__ == "__main__":
    main()