"""Живые сведения о вине с портала «Своё Вино»: народный рейтинг, крепость, подача, гастрономия.

В выданном дампе каталога этих полей нет, а рейтинг к тому же меняется каждый день, так что
в каталог его и не положить. Портал отдаёт карточку по тому же slug открытым JSON-API
(api.vino-svoe.ru/v1/wines/<slug>) — его и спрашиваем, когда интерфейс открывает карточку.

Портал — необязательная добавка: карточка вина строится из каталога и без него. Поэтому
короткий таймаут, ответ в кэше на несколько часов, неудача кэшируется ненадолго (чтобы
лежащий портал не тормозил каждую карточку), и любые сбои превращаются в None, а не в 500.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import quote

log = logging.getLogger("wine_scanner.portal")

API_URL = "https://api.vino-svoe.ru/v1"
SITE_URL = "https://vino-svoe.ru"
TTL = 6 * 3600
MISS_TTL = 10 * 60
TIMEOUT = 3.0
MAX_ITEMS = 5000
_TEMPERATURE = re.compile(r"^\d{1,2}(?:\s*[-–]\s*\d{1,2})?$")


def parse(data: dict, slug: str, site: str = SITE_URL) -> dict:
    """Из ответа портала — только то, что показываем, и только в разумных пределах."""
    rating = data.get("publicRating")
    alcohol = data.get("alcohol")
    temperature = data.get("temperature")
    dishes = data.get("dishes") if isinstance(data.get("dishes"), list) else []
    return {
        "rating": round(float(rating), 2)
        if isinstance(rating, int | float) and 0 < rating <= 5
        else None,
        "alcohol": float(alcohol) if isinstance(alcohol, int | float) and 0 < alcohol < 30 else None,
        "temperature": temperature.replace(" ", "").replace("–", "-")
        if isinstance(temperature, str) and _TEMPERATURE.match(temperature.strip())
        else None,
        "dishes": [d["name"] for d in dishes if isinstance(d, dict) and isinstance(d.get("name"), str)][
            :8
        ],
        "url": f"{site}/wines/{quote(slug, safe='')}",
    }


def _fetch_json(url: str, timeout: float) -> dict | None:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "wine-scanner/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — адрес из настроек
            return json.loads(response.read(512 * 1024))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None  # вина нет на портале — это ответ, а не сбой
        raise


class Portal:
    def __init__(
        self,
        api: str = API_URL,
        site: str = SITE_URL,
        fetch: Callable[[str, float], dict | None] = _fetch_json,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api = api.rstrip("/")
        self.site = site.rstrip("/")
        self.fetch = fetch
        self.clock = clock
        self.cache: dict[str, tuple[float, dict | None]] = {}
        self.lock = threading.Lock()
        self.errors = 0

    @classmethod
    def from_env(cls) -> "Portal | None":
        """WINE_PORTAL=0 выключает обращения к порталу (офлайн-стенд, тесты)."""
        if os.environ.get("WINE_PORTAL", "1") == "0":
            return None
        return cls(
            api=os.environ.get("WINE_PORTAL_API", API_URL),
            site=os.environ.get("WINE_PORTAL_SITE", SITE_URL),
        )

    def facts(self, slug: str) -> dict | None:
        now = self.clock()
        with self.lock:
            hit = self.cache.get(slug)
        if hit and hit[0] > now:
            return hit[1]
        try:
            data = self.fetch(f"{self.api}/wines/{quote(slug, safe='')}", TIMEOUT)
            value = parse(data, slug, self.site) if isinstance(data, dict) else None
            expires = now + TTL
        except Exception as error:  # noqa: BLE001 — портал не должен ронять карточку
            self.errors += 1
            log.warning("портал не ответил по %s: %s", slug, error)
            # Устаревший ответ лучше пустого: рейтинг за полдня почти не меняется.
            value = hit[1] if hit else None
            expires = now + MISS_TTL
        with self.lock:
            if len(self.cache) >= MAX_ITEMS:
                self.cache.clear()
            self.cache[slug] = (expires, value)
        return value
