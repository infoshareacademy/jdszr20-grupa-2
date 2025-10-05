from nicegui import ui
import pandas as pd

from recommend import recommend_by_answers  # zakładam, że plik jest obok apka.py

# ========= USTAWIENIA =========
CSV_PATH = "data/plants_clean_final.csv"

# ========= DANE =========
df = pd.read_csv(CSV_PATH)

# ========= STAN ODPOWIEDZI Z QUIZU =========
answers = {
    "light": None,
    "watering": None,
    "winter_temp": None,
    "summer_heat": None,
    "plant_type": None,
    "pets": None,
    "children": None,
    "tags": [],
}

# ========= MAPOWANIA UI -> WARTOŚCI W CSV =========
LIGHT_MAP = {
    "Mało światła": "Rozproszone światło",
    "Dużo światła": "Bezpośrednie światło",
}
WATERING_MAP = {
    "Rzadko – podlewam dopiero, gdy ziemia jest całkiem sucha.": "Rzadko",
    "Umiarkowanie – podlewam, gdy ziemia przeschnie do połowy lub co kilka dni.": "Umiarkowanie",
    "Często – chcę, aby ziemia była stale wilgotna.": "Często",
}
WINTER_TEMP_MAP = {
    "Może być bardzo zimno (0–5°C)": "Odporna na chłód",
    "Bywa chłodno (6–12°C)": "Toleruje chłód",
    "Zawsze ciepło, powyżej 13°C": "Tylko temperatura pokojowa",
}
SUMMER_HEAT_MAP = {
    "Nie, pozostaje raczej chłodne": "Wrażliwa na upał",
    "Trochę się nagrzewa, ale nie jest duszno": "Normalna tolerancja ciepła",
    "Tak, w upały robi się naprawdę gorąco": "Odporna na upał",
}

# ========= LOGIKA REKOMENDACJI =========
def recommend_plants():
    """Najpierw próbuje modelu, a w razie błędu – fallback po CSV (zawsze coś zwróci)."""

    # Toksyczność: jeśli są zwierzęta lub zaznaczono bezpieczeństwo dla dzieci -> preferuj nietoksyczne
    toxicity_any = 0 if (
        answers.get("pets") == "Tak"
        or answers.get("children") == "Tak → tylko rośliny nietoksyczne / lekko toksyczne"
    ) else 1

    # Przekładamy odpowiedzi z UI na dokładne klucze i wartości z CSV
    answers_mapped = {
        "light_level_clean": LIGHT_MAP.get(answers.get("light")),
        "watering_group": WATERING_MAP.get(answers.get("watering")),
        "tempmin_pasmo": WINTER_TEMP_MAP.get(answers.get("winter_temp")),
        "tempmax_pasmo": SUMMER_HEAT_MAP.get(answers.get("summer_heat")),
        "category_group": answers.get("plant_type"),
        "toxicity_any": toxicity_any,
    }

    # --- 1) Próba użycia modelu ---
    try:
        recs = recommend_by_answers(answers_mapped, top_n=5, feature_csv=CSV_PATH)
        # oczekujemy df z kolumną 'latin' (lub przynajmniej rekordy z kluczem 'latin')
        return recs.to_dict(orient="records")

    # --- 2) Fallback: proste filtrowanie po CSV, gdy model wywali wyjątek ---
    except Exception as e:
        print("Błąd rekomendacji (model):", e)

        out = df.copy()

        # Filtrowanie tylko po podanych kryteriach
        if answers_mapped.get("light_level_clean"):
            out = out[out["light_level_clean"] == answers_mapped["light_level_clean"]]
        if answers_mapped.get("watering_group"):
            out = out[out["watering_group"] == answers_mapped["watering_group"]]
        if answers_mapped.get("tempmin_pasmo"):
            out = out[out["tempmin_pasmo"] == answers_mapped["tempmin_pasmo"]]
        if answers_mapped.get("tempmax_pasmo"):
            out = out[out["tempmax_pasmo"] == answers_mapped["tempmax_pasmo"]]
        if answers_mapped.get("category_group"):
            out = out[out["category_group"] == answers_mapped["category_group"]]
        if answers_mapped.get("toxicity_any") == 0:
            # gdy wymagamy nietoksyczności – trzymaj tylko 0
            out = out[out["toxicity_any"] == 0]

        # Jeśli wyników jest mało, rozluźnij kryteria: trzymaj głównie światło (+ nietoksyczność)
        if len(out) < 5:
            tmp = df.copy()
            if answers_mapped.get("light_level_clean"):
                tmp = tmp[tmp["light_level_clean"] == answers_mapped["light_level_clean"]]
            if answers_mapped.get("toxicity_any") == 0:
                tmp = tmp[tmp["toxicity_any"] == 0]
            out = pd.concat([out, tmp]).drop_duplicates(subset=["latin"])

        # Heurystyczne sortowanie „łatwości” (podlewanie + tolerancje temp, jeśli kolumny istnieją)
        watering_score = out["watering_group"].map({"Rzadko": 2, "Umiarkowanie": 1, "Często": 0}).fillna(0)
        cold = out["is_cold_tolerant"] if "is_cold_tolerant" in out.columns else 0
        heat = out["is_heat_tolerant"] if "is_heat_tolerant" in out.columns else 0
        out = out.assign(_score=watering_score + (cold if hasattr(cold, "fillna") else cold)
                         + (heat if hasattr(heat, "fillna") else heat))
        out = out.sort_values("_score", ascending=False)

        # Zwróć do 5 rekordów w formacie używanym przez UI
        return out[["latin"]].head(5).to_dict(orient="records")

# ========= UI =========
def quiz():
    ui.label("🌱 Quiz: Znajdź idealną roślinę").classes("text-2xl font-bold mb-6")

    with ui.card():
        ui.label("1. Jakie światło jest w miejscu, gdzie planujesz postawić roślinę?")
        ui.radio(["Mało światła", "Dużo światła"], on_change=lambda e: answers.update(light=e.value))

    with ui.card():
        ui.label("2. Jak często chcesz podlewać swoją roślinę?")
        ui.radio([
            "Rzadko – podlewam dopiero, gdy ziemia jest całkiem sucha.",
            "Umiarkowanie – podlewam, gdy ziemia przeschnie do połowy lub co kilka dni.",
            "Często – chcę, aby ziemia była stale wilgotna."
        ], on_change=lambda e: answers.update(watering=e.value))

    with ui.card():
        ui.label("3. Jak chłodno bywa zimą w miejscu, gdzie chcesz postawić roślinę?")
        ui.radio([
            "Może być bardzo zimno (0–5°C)",
            "Bywa chłodno (6–12°C)",
            "Zawsze ciepło, powyżej 13°C"
        ], on_change=lambda e: answers.update(winter_temp=e.value))

    with ui.card():
        ui.label("4. Czy miejsce mocno nagrzewa się latem?")
        ui.radio([
            "Nie, pozostaje raczej chłodne",
            "Trochę się nagrzewa, ale nie jest duszno",
            "Tak, w upały robi się naprawdę gorąco"
        ], on_change=lambda e: answers.update(summer_heat=e.value))

    with ui.card():
        ui.label("5. Jakiego rodzaju rośliny szukasz?")
        ui.radio([
            "Sukulent / kaktus",
            "Palma / drzewko ozdobne",
            "Paproć / roślina zielona",
            "Roślina kwitnąca",
            "Wisząca"
        ], on_change=lambda e: answers.update(plant_type=e.value))

    with ui.card():
        ui.label("6. Czy masz w domu zwierzęta?")
        ui.radio(["Tak", "Nie"], on_change=lambda e: answers.update(pets=e.value))

    with ui.card():
        ui.label("7. Czy w Twoim domu są małe dzieci lub osoby wrażliwe?")
        ui.radio([
            "Tak → tylko rośliny nietoksyczne / lekko toksyczne",
            "Nie → wszystkie kategorie"
        ], on_change=lambda e: answers.update(children=e.value))

    with ui.card():
        ui.label("8. Na koniec – wybierz dodatkowe tagi (opcjonalnie):")
        tags = [
            "#CzystePowietrze", "#BezProblemów", "#PrzyjazneDlaZwierząt",
            "#RoślinnyGigant", "#WisząceStyle", "#KwiatowyPower",
            "#PaprotkaLover", "#MałoWymagające", "#Kolekcjoner"
        ]
        for tag in tags:
            ui.checkbox(tag, on_change=lambda e, t=tag: (
                answers["tags"].append(t) if e.value else answers["tags"].remove(t)
            ))

    ui.button("📊 Pokaż rekomendacje", on_click=show_results).classes("mt-6 bg-green-600 text-white")

def show_results():
    recommendations = recommend_plants()

    with ui.dialog() as dialog, ui.card():
        ui.label("🌿 Rośliny, które mogą Cię zainteresować:").classes("text-xl font-bold mb-4")

        if not recommendations:
            ui.label("Brak wyników — uzupełnij odpowiedzi lub spróbuj ponownie.")
        else:
            for plant in recommendations:
                # z modelu lub fallbacku – w obu przypadkach próbujemy pobrać 'latin'
                name = plant.get("latin") if isinstance(plant, dict) else str(plant)
                ui.label(f"- {name}")

        if answers["tags"]:
            ui.label(f"Wybrane tagi: {', '.join(answers['tags'])}")

        ui.button("Zamknij", on_click=dialog.close)

    dialog.open()

# ========= START =========
quiz()
ui.run()
