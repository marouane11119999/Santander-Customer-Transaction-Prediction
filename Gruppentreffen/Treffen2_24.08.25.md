
### 1. Austausch über Erkenntnisse aus der EDA

- Der Datensatz ist **stark unbalanciert**. 

- Viele Features sehen **annähernd normalverteilt** aus.

- Features sind **weitgehend unkorreliert**.
    
- Feature mit **Standardabweichung nahe 0** (fast keine Varianz).
    
- **Keine fehlenden Werte** im Datensatz vorhanden.
    
- Auffällig: **Sehr wenig einzigartige Werte pro Spalte**

### 2. Arbeitsaufteilung

- **Zenep** → Vertiefte EDA
    
- **Marouane** → Modellierung mit **XGBoost**
    
- **Hannah** → Modellierung mit **Neuronalen Netzen**
    
- **Jimmy** → Modellierung mit **LightGBM (alternativ KNN)**


### 3. Train-/Test-Split

Eine einheitliche Funktion schreiben und im Git bereitstellen, damit wir alle denselben Split verwenden und die Modelle fair vergleichen können.

### 4. Baseline-Modell

Als gemeinsames Baseline-Modell verwenden wir eine logistische Regression. Jedes weitere Modell wird zunächst mit der LR verglichen und am Ende vergleichen wir zusätzlich unsere Modelle untereinander.


### 5. Nächste Schritte

- Erkenntnisse aus der EDA für **Feature Engineering & Feature Selection** nutzen.
    
- Beginn mit den ersten Modell-Trainings.
    
- Austausch über erste Ergebnisse bei Bedarf schon vor dem nächsten Treffen.
    

### 6. Nächstes Treffen

- Regulär in **2 Wochen am 07.09.**
    
- Frühere Abstimmung möglich, falls es schon vorher Gesprächsbedarf gibt.    