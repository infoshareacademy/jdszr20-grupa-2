from nicegui import ui
import pandas as pd

# --- wczytanie danych ---
df = pd.read_csv("house_plants_features.csv")

# --- odpowiedzi użytkownika ---
answers = {
    "light": None,
    "watering": None,
    "winter_temp": None,
    "summer_heat": None,
    "plant_type": None,
    "pets": None,
    "children": None,
    "tags": []
}

# --- dopasowywanie roślin ---
def recommend_plants():
    filtered = df.copy()

    # filtrowanie światła
    if answers["light"] and "light" in filtered.columns:
        filtered = filtered[filtered["light"] == answers["light"]]

    # podlewanie
    if answers["watering"] and "watering" in filtered.columns:
        filtered = filtered[filtered["watering"] == answers["watering"]]

    # zimowa temp
    if answers["winter_temp"] and "winter_temp" in filtered.columns:
        filtered = filtered[filtered["winter_temp"] == answers["winter_temp"]]

    # letnie ciepło
    if answers["summer_heat"] and "summer_heat" in filtered.columns:
        filtered = filtered[filtered["summer_heat"] == answers["summer_heat"]]

    # typ rośliny
    if answers["plant_type"] and "plant_type" in filtered.columns:
        filtered = filtered[filtered["plant_type"] == answers["plant_type"]]

    # zwierzęta
    if answers["pets"] == "Tak" and "toxic_to_pets" in filtered.columns:
        filtered = filtered[filtered["toxic_to_pets"] == False]

    # dzieci i osoby wrażliwe
    if answers["children"] and "toxicity_level" in filtered.columns:
        if "Tak" in answers["children"]:
            filtered = filtered[filtered["toxicity_level"].isin(["non-toxic", "slightly toxic"])]

    # jeśli brak wyników → fallback
    if filtered.empty:
        filtered = df

    # wybierz max 3 losowe rośliny
    result = filtered.sample(min(3, len(filtered))).to_dict(orient="records")
    return result


# --- quiz ---
def quiz():
    ui.label("🌱 Quiz: Znajdź idealną roślinę").classes("text-2xl font-bold mb-6")

    # 1. światło
    with ui.card():
        ui.label("1. Jakie światło jest w miejscu, gdzie planujesz postawić roślinę?")
        ui.radio(["Mało światła", "Dużo światła"], on_change=lambda e: answers.update(light=e.value))

    # 2. podlewanie
    with ui.card():
        ui.label("2. Jak często chcesz podlewać swoją roślinę?")
        ui.radio([
            "Rzadko – podlewam dopiero, gdy ziemia jest całkiem sucha.",
            "Umiarkowanie – podlewam, gdy ziemia przeschnie do połowy lub co kilka dni.",
            "Często – chcę, aby ziemia była stale wilgotna."
        ], on_change=lambda e: answers.update(watering=e.value))

    # 3. zima
    with ui.card():
        ui.label("3. Jak chłodno bywa zimą w miejscu, gdzie chcesz postawić roślinę?")
        ui.radio([
            "Może być bardzo zimno (0–5°C)",
            "Bywa chłodno (6–12°C)",
            "Zawsze ciepło, powyżej 13°C"
        ], on_change=lambda e: answers.update(winter_temp=e.value))

    # 4. lato
    with ui.card():
        ui.label("4. Czy miejsce mocno nagrzewa się latem?")
        ui.radio([
            "Nie, pozostaje raczej chłodne",
            "Trochę się nagrzewa, ale nie jest duszno",
            "Tak, w upały robi się naprawdę gorąco"
        ], on_change=lambda e: answers.update(summer_heat=e.value))

    # 5. typ rośliny
    with ui.card():
        ui.label("5. Jakiego rodzaju rośliny szukasz?")
        ui.radio([
            "Sukulent / kaktus",
            "Palma / drzewko ozdobne",
            "Paproć / roślina zielona",
            "Roślina kwitnąca",
            "Wisząca"
        ], on_change=lambda e: answers.update(plant_type=e.value))

    # 6. zwierzęta
    with ui.card():
        ui.label("6. Czy masz w domu zwierzęta?")
        ui.radio(["Tak", "Nie"], on_change=lambda e: answers.update(pets=e.value))

    # 7. dzieci
    with ui.card():
        ui.label("7. Czy w Twoim domu są małe dzieci lub osoby wrażliwe?")
        ui.radio([
            "Tak → tylko rośliny nietoksyczne / lekko toksyczne",
            "Nie → wszystkie kategorie"
        ], on_change=lambda e: answers.update(children=e.value))

    # 8. tagi
    with ui.card():
        ui.label("8. Na koniec – wybierz dodatkowe tagi:")
        tags = [
            "#CzystePowietrze", "#BezProblemów", "#PrzyjazneDlaZwierząt",
            "#RoślinnyGigant", "#WisząceStyle", "#KwiatowyPower",
            "#PaprotkaLover", "#MałoWymagające", "#Kolekcjoner"
        ]
        for tag in tags:
            ui.checkbox(tag, on_change=lambda e, t=tag: (
                answers["tags"].append(t) if e.value else answers["tags"].remove(t)
            ))

    # przycisk końcowy
    ui.button("📊 Pokaż rekomendacje", on_click=show_results).classes("mt-6 bg-green-600 text-white")


def show_results():
    recommendations = recommend_plants()

    # sprawdź czy istnieje kolumna 'latin' lub 'Latin'
    name_column = None
    for col in ["latin", "Latin", "latin_name"]:
        if col in df.columns:
            name_column = col
            break

    with ui.dialog() as dialog, ui.card():
        ui.label("🌿 Rośliny, które mogą Cię zainteresować:").classes("text-xl font-bold mb-4")
        for plant in recommendations:
            if name_column:
                ui.label(f"- {plant.get(name_column, 'Nieznana roślina')}")
            else:
                ui.label(str(plant))  # pokaż cały rekord jeśli brak kolumny
        if answers["tags"]:
            ui.label(f"Wybrane tagi: {', '.join(answers['tags'])}")
        ui.button("Zamknij", on_click=dialog.close)
    dialog.open()


# --- start ---
quiz()
ui.run()
