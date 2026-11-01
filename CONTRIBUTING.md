# Zmiany w projekcie

- Zachowuj trzy opisane w README ścieżki. Nowy eksperyment rozwijaj na osobnej gałęzi.
- Nie zapisuj datasetu, wag, cache ani wyników w Git.
- Nowe treningi uruchamiaj pod nową nazwą; rozróżniaj inicjalizację wag od pełnego wznowienia.
- Nie używaj obrazu docelowego do budowania warunków generowania z EEG.
- Dokumentuj komendę, podział danych i checkpoint użyty do wyniku.
- Przed oddaniem zmian: `python -m compileall -q eeg_mae` i `ruff format --check eeg_mae`.
