# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (base)
#     language: python
#     name: base
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/Alice-Ferri/data-intensive-project-2026/blob/main/data-intensive-project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Determinare la categoria della malattia della cellula.

# %% [markdown]
# Programmazione di data intensive a.a. 2025/2026
#
# Alice Ferri, alice.ferri8@studio.unibo.it
#
# Davide rossi, davide.rossi47@studio.unibo.it

# %% [markdown]
# # Parte 1 - Descrizione del contesto del problema

# %% [markdown]
# Il dataset di riferimento è [rilevamento di anomalie di cellule del sangue](https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025/data) presente in Kaggle.
#
# Il dataset contiene le informazioni di cellule del sangue sane e malate.
# Tali dati permettono di suddividere le cellule in 7 categorie, 5 di queste classificate come anomale e 2 come normali.
#
# L'obbiettivo del progetto è sviluppare un classificatore che sia in grando di determinare la categoria della cellula.

# %% [markdown]
# ### Caricamento librerie

# %% [markdown]
# Installiamo nel kernel la libreria per importare il dataset da kagglehub

# %%
pip install kagglehub[pandas-datasets]

# %% [markdown]
# Importiamo le librerie che utilizzeremo

# %%
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter

# %% [markdown]
# ### Caricamento dei dati e preprocessing

# %% [markdown]
# Carichiamo il dataset in un pandas dataframe da kagglehub

# %%
# nome del file del dataset
file_path = "blood_cell_anomaly_detection.csv"

data_raw = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "alitaqishah/blood-cell-anomaly-detection-2025",
  file_path
)

data_raw.tail()

# %% [markdown]
# Con il metodo info stampiamo le informazioni principali come numero di istanze, data type
# per feature, e spazio in memoria

# %%
data_raw.info(memory_usage="deep")

# %% [markdown]
# Inoltre si può notare che non sono presenti valori nulli nel dataset.

# %% [markdown]
# ### Significato delle features
# Il dataset contiene 36 features, di seguito sono riportate raggruppate per tipologia:

# %% [markdown]
# **Morphology** — diametro, circolarità, eccentricità, lobularità, granularità, area del nucleo, densità della cromatina
#
# **Color** — valori RGB medi, intensità della colorazione
#
# **Clinical CBC** — dati ricavati dagli esami del sangue, come quantità globuli bianchi e rossi, piastrine, emoglobina etc.
#
# **Acquisition** — dati riguardanti l'immagine al microscopio, come il modello, la risoluzione e grado di ingrandimento
#
# **AI Scores** — dati legati al modello CytoDiffusion che risolve la stessa tipologia di problema, come confidenza di anomalia della cellula e di suddivisione del tipo della cellula. Troviamo anche il valore di sicurezza della stima del tipo di cellula di un medico esperto

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# La variabile che tenteremo di predirre è **disease_category**. Le label possono essere:
# - __Normal_WBC__, globuli bianchi normali
# - __Normal_RBC__, globuli rossi normali
# - __Leukemia__
# - __Anemia__
# - __Sickle_Cell_Anemia__, anemia falciforme
# - __Infection__
# - __Artefact__
#

# %% [markdown]
# ### Scrematura dei dati

# %% [markdown]
# Analizzando il dataset si notano le features _cell_id_ che risulta essere un identificatore del record e _dataset_source_ che contiene il database di riferimento dell'informazione.
#
# Sono variabili non informative per la nostra indagine, si decide così di rimuoverle dal nostro dataset.

# %% [markdown]
# Si notano inoltre le features legate all'AI scores _cytodiffusion_anomaly_score_  _cytodiffusion_classification_confidence_, come descritte in precedenze. Queste features sono dei punteggi di confidenza di un modello già esistente allenato sulle stesse cellule di riferimento ed in cui è integrata anche l'analisi delle immagini.
# E' presente anche la variabile _labeller_confidence_score_ che rappresenta la stima di confidenza dell' umano esperto.
#
# Si è deciso di rimuovere queste features perchè sono dati che non costituiscono l'input del nostro problema.
# Infatti il modello si aspetta di ricevere informazioni legate esclusivamente alla cellula e non dati legati ad altre rilevazioni, risulterebbe influente per l'apprendimento del nostro modello e altererebbero la considerazioni di altre variabili importanti. In un caso reale il modello non avrebbe a disposizione queste informazioni.

# %% [markdown]
# Si preserva anche il dataset nella forma originale con le suddette features per utilizzarle nella fase di valutazione dei modelli sviluppati.

# %%
ds = data_raw.copy()

ds.drop(columns=['cell_id',
                 'dataset_source',
                 'cytodiffusion_anomaly_score',
                 'cytodiffusion_classification_confidence',
                 'labeller_confidence_score'], inplace=True)

# %%
ds.head(10)

# %% [markdown]
# # Parte 2 - Esplorazione dei dati

# %%
counts = ds['disease_category'].value_counts()
num_categorie = len(counts)

colori = plt.cm.tab10(range(num_categorie)) 

plt.bar(counts.index, counts.values, color=colori)
plt.title('Disease category')
plt.xlabel('Categorie')
plt.ylabel('Conteggio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Osservando il grafico a barre è evidente che la varabile target _disease_category_ risulta essere poco bilanciata, poichè il numero di istanze di tipo Normal_WBC è molto più significativa rispetto alle restanti categorie. Il problema in esame quindi risulta non essere bilanciato, si proverà quindi ad applicare tecniche di bilanciamento come oversampling o undersampling

# %%
ds['anomaly_label'].value_counts().plot.pie(autopct='%1.1f%%')

# %% [markdown]
# ## Analisi statistiche e valori outlier

# %% [markdown]
# Stampiamo le statistiche per le feature del dataset

# %%
pd.set_option('display.max_columns', None)
ds.describe()

# %%
ds.iloc[ds.cell_area_px > ds.image_resolution_px**2].sort_values(by="cell_area_px")

# %% [markdown]
# Come si osserva dalla descrizione del dataset, si rilevano media, valore massimo, valore minimo, deviazione standard e percentili di ogni features.
# Notiamo già che non sono presenti valori di massimi e minimi fuori da range accettabili per nessuna delle feature

# %% [markdown]
# Si visualizzano le distribuzioni di tutte le variabili continue del dataset, escludendo quelle non legate ai dati della cellula e in cui l'unità di misura è un punteggio attribuito su una scala di valori possibili. 

# %%
plt.figure(figsize=(25, 20)) 

plt.subplot(3, 3, 1)
plt.title('Cell Diamater')
plt.boxplot(ds['cell_diameter_um'])
plt.ylabel('um')

plt.subplot(3, 3, 2)
plt.title('Nucleus Area')
plt.boxplot(ds['nucleus_area_pct'])
plt.ylabel('percentage')

plt.subplot(3, 3, 3)
plt.title('White Blood Cell')
plt.boxplot(ds['wbc_count_per_ul'])
plt.ylabel('count per microlitre')

plt.subplot(3, 3, 4)
plt.title('Red Blood Cell')
plt.boxplot(ds['rbc_count_millions_per_ul'])
plt.ylabel('count in millions per microlitre')

plt.subplot(3, 3, 5)
plt.title('Hemoglobin')
plt.boxplot(ds['hemoglobin_g_dl'])
plt.ylabel('g/dL')

plt.subplot(3, 3, 6)
plt.title('Hematocrit')
plt.boxplot(ds['hematocrit_pct'])
plt.ylabel('percentage')

plt.subplot(3, 3, 7)
plt.title('Platelet')
plt.boxplot(ds['platelet_count_per_ul'])
plt.ylabel('count per microlitre')

plt.subplot(3, 3, 8)
plt.title('Mean Corpuscular Volume')
plt.boxplot(ds['mcv_fl'])
plt.ylabel('percentage')

plt.subplot(3, 3, 9)
plt.title('Mean corpuscular Haemoglobin')
plt.boxplot(ds['mchc_g_dl'])
plt.ylabel('g/dL')

plt.tight_layout()
plt.show()

# %% [markdown]
# Dai grafici, in particolare i boxplot, e dalla tabella generata dal metodo describe si possono notare alcuni valori outliers per certe feature. 
# Tuttavia a seguito di analisi e ricerche si è appurato che tali valori sono accettabili e possono ricondursi a casistiche reali. È stato verificato che non vi sia dunque la presenza di valori completamente errati. 
# Inoltre, per alcune feature i valori outlier sono proprio indicativi di anomalie delle cellule. 
# Ad esempio nel caso dell'emoglobina valori bassi, che quindi si discostano tanto dalla media, sono indicatori di anemia

# %%
plt.figure(figsize=(25, 20)) 

plt.subplot(3, 3, 1)
plt.title('Cell Diamater')
plt.hist(ds['cell_diameter_um'])
plt.xlabel('um')

plt.subplot(3, 3, 2)
plt.title('Nucleus Area')
plt.hist(ds['nucleus_area_pct'])
plt.xlabel('percentage')

plt.subplot(3, 3, 3)
plt.title('White Blood Cell')
plt.hist(ds['wbc_count_per_ul'])
plt.xlabel('count per microlitre')

plt.subplot(3, 3, 4)
plt.title('Red Blood Cell')
plt.hist(ds['rbc_count_millions_per_ul'])
plt.xlabel('count in millions per microlitre')

plt.subplot(3, 3, 5)
plt.title('Hemoglobin')
plt.hist(ds['hemoglobin_g_dl'])
plt.xlabel('g/dL')

plt.subplot(3, 3, 6)
plt.title('Hematocrit')
plt.hist(ds['hematocrit_pct'])
plt.xlabel('percentage')

plt.subplot(3, 3, 7)
plt.title('Platelet')
plt.hist(ds['platelet_count_per_ul'])
plt.xlabel('count per microlitre')

plt.subplot(3, 3, 8)
plt.title('Mean Corpuscular Volume')
plt.hist(ds['mcv_fl'])
plt.xlabel('percentage')

plt.subplot(3, 3, 9)
plt.title('Mean corpuscular Haemoglobin')
plt.hist(ds['mchc_g_dl'])
plt.xlabel('g/dL')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Relazione tra variabili

# %%

# %% [markdown]
# # in relazione variabili
# # cell_area_px e permiter_px da aggiungere?
# # bilanciamento
