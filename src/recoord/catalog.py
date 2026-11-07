from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .table import Table

CATEGORIES = {
    "beverage": ("Mineral water", "Sparkling water", "Iced tea", "Cola", "Orange juice"),
    "dairy": ("Whole milk", "Greek yoghurt", "Butter", "Cream cheese"),
    "pantry": ("Olive oil", "Tomato passata", "Rice", "Wheat flour"),
    "care": ("Shower gel", "Shampoo", "Hand soap", "Toothpaste"),
}
LOCALE = {
    "beverage": "Getraenke", "dairy": "Milchprodukte",
    "pantry": "Vorratsschrank", "care": "Koerperpflege",
}
UNIT_VOLUMES = (200, 330, 500, 750, 1000, 1500)
PACK_SIZES = (1, 4, 6, 12)
COLUMNS = ["gtin", "vendor", "brand", "title", "category", "description",
           "unit_volume_ml", "pack_size", "total_volume_ml", "price_cents",
           "currency", "country"]


@dataclass
class CatalogSpec:
    products: int = 300
    vendors: int = 18
    offers: int = 1_200
    seed: int = 0


def build(spec: CatalogSpec) -> Table:
    rng = random.Random(spec.seed)
    brands = [f"Brand{chr(65 + i)}" for i in range(12)]
    vendors = [f"v{i:02d}" for i in range(spec.vendors)]
    styles = {v: rng.choice(("plain", "german", "upper", "verbose")) for v in vendors}
    countries = {v: rng.choice(("DE", "AT", "NL", "FR", "GB")) for v in vendors}

    products = []
    for i in range(spec.products):
        category = rng.choice(list(CATEGORIES))
        products.append({
            "gtin": f"4{i:012d}",
            "brand": rng.choice(brands),
            "name": rng.choice(CATEGORIES[category]),
            "category": category,
            "unit_volume_ml": rng.choice(UNIT_VOLUMES),
            "pack_size": rng.choice(PACK_SIZES),
            "price": rng.randrange(89, 2_499),
        })

    rows = []
    for offer in range(spec.offers):
        product = products[offer % len(products)]

        vendor = vendors[offer % len(vendors)] if offer < len(vendors) else rng.choice(vendors)
        total = product["unit_volume_ml"] * product["pack_size"]
        rows.append({
            "id": f"o{offer:06d}",
            "gtin": product["gtin"],
            "vendor": vendor,
            "brand": product["brand"],
            "title": _title(product, styles[vendor]),
            "category": product["category"],
            "description": _description(product, styles[vendor]),
            "unit_volume_ml": str(product["unit_volume_ml"]),
            "pack_size": str(product["pack_size"]),
            "total_volume_ml": str(total),
            "price_cents": str(product["price"]),
            "currency": "EUR" if countries[vendor] != "GB" else "GBP",
            "country": countries[vendor],
        })
    return Table(rows, "id", COLUMNS)


def _title(product: dict[str, Any], style: str) -> str:
    base = f"{product['brand']} {product['name']}"
    if style == "upper":
        return base.upper()
    if style == "german":
        return f"{base} {product['pack_size']}er-Pack"
    if style == "verbose":
        return f"{base} {product['pack_size']} x {product['unit_volume_ml']} ml"
    return base


def _description(product: dict[str, Any], style: str) -> str:
    litres = product["unit_volume_ml"] / 1000
    if style == "german":
        return f"{product['name']}, {product['pack_size']}er-Pack, {str(litres).replace('.', ',')} L je Flasche"
    if style == "verbose":
        return f"{product['name']} pack of {product['pack_size']}, {product['unit_volume_ml']} ml per item"
    return f"{product['name']} {product['pack_size']} x {product['unit_volume_ml']} ml"


def private_tables(seed: int = 0) -> dict[str, dict[str, str]]:
    return {
        "locale_category": {v.lower(): k for k, v in LOCALE.items()},
        "currency": {"de": "EUR", "at": "EUR", "nl": "EUR", "fr": "EUR", "gb": "GBP"},
    }


def locale_map() -> dict[str, str]:
    return dict(LOCALE)
