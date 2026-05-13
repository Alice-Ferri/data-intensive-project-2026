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
# ### Parte 1 - Descrizione del contesto del problema

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
