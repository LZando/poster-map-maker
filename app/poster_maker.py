import matplotlib
matplotlib.use("Agg")  # IMPORTANT: server-side rendering (Flask/thread-safe)

import time
import json
import pickle
from hashlib import md5
from pathlib import Path
from datetime import datetime

import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.font_manager import FontProperties
from geopy.geocoders import Nominatim


# -----------------------------
# Directories
# -----------------------------
# -----------------------------
# Directories
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = PROJECT_ROOT / "app" / "static" / "themes"
POSTERS_DIR = PROJECT_ROOT / "posters"
POSTERS_DIR.mkdir(exist_ok=True)

FONTS_DIR = PROJECT_ROOT / "app" / "static" / "fonts"


# -----------------------------
# Cache helpers (pickle)
# -----------------------------
class CacheError(Exception):
    pass


def _get_cache_path(key: str, subfolder: str = None) -> Path:
    if subfolder:
        folder = CACHE_DIR / subfolder.replace("|", "_").replace(":", "_").replace("/", "_").replace(" ", "_")
        folder.mkdir(exist_ok=True)
        # Remove subfolder info from key to avoid redundancy in filename if desired,
        # but for now let's just sanitize the key for the filename.
        filename = key.replace("|", "_").replace(":", "_").replace("/", "_").replace(" ", "_") + ".pkl"
        return folder / filename
    else:
        filename = key.replace("|", "_").replace(":", "_").replace("/", "_").replace(" ", "_") + ".pkl"
        return CACHE_DIR / filename


def cache_get(key: str, subfolder: str = None):
    p = _get_cache_path(key, subfolder)
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return None


def cache_set(key: str, obj, subfolder: str = None) -> None:
    p = _get_cache_path(key, subfolder)
    try:
        with p.open("wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        # Fallback to MD5 if filename is too long or has issues
        hash_name = f"{md5(key.encode('utf-8')).hexdigest()}.pkl"
        folder = p.parent
        p = folder / hash_name
        with p.open("wb") as f:
            pickle.dump(obj, f)


# -----------------------------
# Coordinates cache (json)
# -----------------------------
def load_coordinates_cache(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_coordinates_cache(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -----------------------------
# Themes
# -----------------------------
DEFAULT_THEME = {
    "name": "Feature-Based Shading",
    "bg": "#FFFFFF",
    "text": "#000000",
    "gradient_color": "#FFFFFF",
    "water": "#C0C0C0",
    "parks": "#F0F0F0",
    "road_motorway": "#0A0A0A",
    "road_primary": "#1A1A1A",
    "road_secondary": "#2A2A2A",
    "road_tertiary": "#3A3A3A",
    "road_residential": "#4A4A4A",
    "road_default": "#3A3A3A"
}


def load_theme(theme_name: str) -> dict:
    if not theme_name:
        return DEFAULT_THEME
        
    # Robust loading: lower case and underscores
    clean_name = theme_name.lower().strip().replace(" ", "_")
    theme_file = THEMES_DIR / f"{clean_name}.json"
    
    if not theme_file.exists():
        # Try original name as fallback
        theme_file = THEMES_DIR / f"{theme_name}.json"
        if not theme_file.exists():
            return DEFAULT_THEME

    with theme_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # merge with defaults so missing keys don’t break rendering
    merged = dict(DEFAULT_THEME)
    merged.update(data)
    return merged


# -----------------------------
# Fonts (optional Roboto)
# -----------------------------
def load_fonts():
    fonts = {
        "bold": FONTS_DIR / "Roboto-Bold.ttf",
        "regular": FONTS_DIR / "Roboto-Regular.ttf",
        "light": FONTS_DIR / "Roboto-Light.ttf",
    }
    for k, p in fonts.items():
        if not p.exists():
            return None
    return {k: str(p) for k, p in fonts.items()}


FONTS = load_fonts()


# -----------------------------
# Styling helpers
# -----------------------------
def create_gradient_fade(ax, color: str, location: str = "bottom", zorder: int = 10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    rgba = np.zeros((256, 4))
    rgba[:, 0], rgba[:, 1], rgba[:, 2] = rgb

    if location == "bottom":
        rgba[:, 3] = np.linspace(1, 0, 256)
        y0, y1 = 0.0, 0.25
    else:
        rgba[:, 3] = np.linspace(0, 1, 256)
        y0, y1 = 0.75, 1.0

    cmap = mcolors.ListedColormap(rgba)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    yr = ylim[1] - ylim[0]
    y_bottom = ylim[0] + yr * y0
    y_top = ylim[0] + yr * y1

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=cmap,
        zorder=zorder,
        origin="lower",
    )


def get_edge_colors_by_type(G, theme: dict):
    colors = []
    for _, _, data in G.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"

        if highway in ["motorway", "motorway_link"]:
            c = theme["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            c = theme["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            c = theme["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            c = theme["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            c = theme["road_residential"]
        else:
            c = theme["road_default"]
        colors.append(c)
    return colors


def get_edge_widths_by_type(G):
    widths = []
    for _, _, data in G.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"

        if highway in ["motorway", "motorway_link"]:
            w = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            w = 1.0
        elif highway in ["secondary", "secondary_link"]:
            w = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            w = 0.6
        else:
            w = 0.4
        widths.append(w)
    return widths


# -----------------------------
# Core classes
# -----------------------------
class Poster:
    def __init__(self, city: str, country: str, style: str):
        self.city = city
        self.country = country
        self.style = style  # used as theme name
        self.geolocator = Nominatim(user_agent="city_map_poster", timeout=10)

        self.coords_cache_file = CACHE_DIR / "coordinates.json"
        self._coords = None  # in-memory cache

    def extract_coordinates(self) -> tuple[float, float]:
        if self._coords:
            return self._coords

        key = f"{self.city.lower()}|{self.country.lower()}"
        cache = load_coordinates_cache(self.coords_cache_file)

        if key in cache:
            lat, lon = cache[key]["lat"], cache[key]["lon"]
            self._coords = (lat, lon)
            return self._coords

        query = f"{self.city}, {self.country}"
        time.sleep(1)  # respect Nominatim usage policy
        location = self.geolocator.geocode(query)

        if not location:
            raise ValueError(f"Luogo non trovato: {query}")

        lat, lon = location.latitude, location.longitude
        cache[key] = {
            "lat": lat, 
            "lon": lon, 
            "display_name": location.address,
            "updated_at": datetime.now().isoformat()
        }
        save_coordinates_cache(self.coords_cache_file, cache)

        self._coords = (lat, lon)
        return self._coords


class PosterGeoData(Poster):
    def extract_street_network(self, dist: int = 10000):
        lat, lon = self.extract_coordinates()
        subfolder = f"{lat:.4f}_{lon:.4f}"
        key = f"graph_{dist}"
        cached = cache_get(key, subfolder=subfolder)
        if cached is not None:
            return cached

        G = ox.graph_from_point(
            (lat, lon),
            dist=dist,
            dist_type="bbox",
            network_type="all",
            simplify=True
        )
        time.sleep(0.5)
        cache_set(key, G, subfolder=subfolder)
        return G

    def extract_features(self, tags: dict, dist: int = 10000, name: str = "features"):
        lat, lon = self.extract_coordinates()
        subfolder = f"{lat:.4f}_{lon:.4f}"
        tag_str = "_".join(sorted(tags.keys()))
        key = f"{name}_{dist}_{tag_str}"
        cached = cache_get(key, subfolder=subfolder)
        if cached is not None:
            return cached

        gdf = ox.features_from_point((lat, lon), tags=tags, dist=dist)
        time.sleep(0.3)
        cache_set(key, gdf, subfolder=subfolder)
        return gdf


class PosterRenderer:
    def __init__(self, graph, city: str, country: str, coords: tuple[float, float], theme: dict):
        self.G = graph
        self.city = city
        self.country = country
        self.coords = coords
        self.theme = theme

    def render(self, output_path: str, parks=None, water=None, fmt: str = "png"):
        fig, ax = plt.subplots(figsize=(12, 16), facecolor=self.theme["bg"])
        ax.set_facecolor(self.theme["bg"])
        ax.set_position([0, 0, 1, 1])

        # --- Layer: water polygons ---
        if water is not None and not water.empty:
            water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not water_polys.empty:
                water_polys.plot(ax=ax, facecolor=self.theme["water"], edgecolor="none", zorder=1)

        # --- Layer: parks polygons ---
        if parks is not None and not parks.empty:
            parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not parks_polys.empty:
                parks_polys.plot(ax=ax, facecolor=self.theme["parks"], edgecolor="none", zorder=2)

        # --- Layer: roads with hierarchy coloring ---
        edge_colors = get_edge_colors_by_type(self.G, self.theme)
        edge_widths = get_edge_widths_by_type(self.G)

        ox.plot_graph(
            self.G,
            ax=ax,
            bgcolor=self.theme["bg"],
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=edge_widths,
            show=False,
            close=False
        )

        # --- Gradients ---
        create_gradient_fade(ax, self.theme["gradient_color"], location="bottom", zorder=10)
        create_gradient_fade(ax, self.theme["gradient_color"], location="top", zorder=10)

        # --- Typography ---
        if FONTS:
            font_sub = FontProperties(fname=FONTS["light"], size=22)
            font_coords = FontProperties(fname=FONTS["regular"], size=14)
            base_main = 60
            font_attr = FontProperties(fname=FONTS["light"], size=8)
        else:
            font_sub = FontProperties(family="monospace", size=22)
            font_coords = FontProperties(family="monospace", size=14)
            base_main = 60
            font_attr = FontProperties(family="monospace", size=8)

        # dynamic city font size
        city_len = len(self.city)
        if city_len > 10:
            scale = 10 / city_len
            main_size = max(base_main * scale, 24)
        else:
            main_size = base_main

        if FONTS:
            font_main = FontProperties(fname=FONTS["bold"], size=main_size)
        else:
            font_main = FontProperties(family="monospace", weight="bold", size=main_size)

        spaced_city = "  ".join(list(self.city.upper()))
        ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
                color=self.theme["text"], ha="center", fontproperties=font_main, zorder=11)

        ax.text(0.5, 0.10, self.country.upper(), transform=ax.transAxes,
                color=self.theme["text"], ha="center", fontproperties=font_sub, zorder=11)

        lat, lon = self.coords
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        coords_str = f"{abs(lat):.4f}° {ns} / {abs(lon):.4f}° {ew}"

        ax.text(0.5, 0.07, coords_str, transform=ax.transAxes,
                color=self.theme["text"], alpha=0.7, ha="center",
                fontproperties=font_coords, zorder=11)

        ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
                color=self.theme["text"], linewidth=1, zorder=11)

        ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
                color=self.theme["text"], alpha=0.5, ha="right", va="bottom",
                fontproperties=font_attr, zorder=11)

        save_kwargs = dict(facecolor=self.theme["bg"], bbox_inches="tight", pad_inches=0.05)
        if fmt.lower() == "png":
            save_kwargs["dpi"] = 300

        plt.savefig(output_path, format=fmt.lower(), **save_kwargs)
        plt.close()
        return output_path


# -----------------------------
# Pipeline: generate image
# -----------------------------
def generate_geo_data(city: str, country: str, style: str, dist: int = 12000, fmt: str = "png") -> str:
    poster = PosterGeoData(city, country, style)

    # ALWAYS extract coordinates first to ensure they are cached in coordinates.json
    coords = poster.extract_coordinates()

    # Deterministic output path to allow file-level caching
    city_slug = city.lower().replace(" ", "_")
    country_slug = country.lower().replace(" ", "_")
    output_filename = f"{city_slug}_{country_slug}_{style}_{dist}.{fmt.lower()}"
    output_path = POSTERS_DIR / output_filename
    
    # Check if we already have this specific poster generated
    if output_path.exists():
        return str(output_path)

    G = poster.extract_street_network(dist=dist)

    # optional layers
    parks = poster.extract_features(tags={"leisure": "park", "landuse": "grass"}, dist=dist, name="parks")
    water = poster.extract_features(tags={"natural": "water", "waterway": "riverbank"}, dist=dist, name="water")

    theme = load_theme(style)

    renderer = PosterRenderer(G, city, country, coords, theme)
    return renderer.render(str(output_path), parks=parks, water=water, fmt=fmt)
