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
import seaborn as sns
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

# %% [markdown]
# ## Analisi statistiche e valori outlier

# %% [markdown]
# Stampiamo le statistiche per le feature del dataset

# %%
pd.set_option('display.max_columns', None)
ds.describe()

# %% [markdown]
# Come si osserva dalla descrizione del dataset, si rilevano media, valore massimo, valore minimo, deviazione standard e percentili di ogni features.
# Notiamo già che non sono presenti valori di massimi e minimi fuori da range accettabili per nessuna delle feature

# %% [markdown]
# Dopo una prima analisi dei valori sembrerebbe ci sia una discordanza fra
# i valori delle feature `cell_area_px` e `image_resolution` in quanto il valore massimo della prima è maggiore del valore massimo della seconda. Queste feature rappresentano per ogni cellula i dati dell'immagine di acquisizione. Si sospetta che la feature cell_area_px possa contenere dei valori outlier
#
# Si selezionano i record dove `cell_area_px > image_resolution` e si nota che questa presunta anomalia è in realtà molto ricorrente nel dataset. Si deduce che la feature image_resolution non indichi il numero totale di pixel nell'immagine bensì la dimensione su un asse dell'immagine e che invece i pixel totali siano `image_resolution^2` . Si verifica che non ci siano cellule con area in px maggiore dell'area totale dell'immagine 

# %%
ds.iloc[ds.cell_area_px > ds.image_resolution_px].sort_values(by="cell_area_px")

# %%
ds.iloc[ds.cell_area_px > ds.image_resolution_px**2]

# %% [markdown]
# ### Distribuzioni

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
# Dai grafici, in particolare i boxplot, e dalla tabella generata dal metodo _describe()_ si possono notare alcuni valori outliers per certe features. 
# A seguito di analisi e ricerche si è appurato che tali valori sono accettabili e possono ricondursi a casistiche reali. È stato verificato che non vi sia dunque la presenza di valori completamente errati.
#
# Per le features _cell_diameter_um_ e _platelet_count_per_ul_ l'analisi ha evidenziato per ciascuna un singolo elemento outlier che si discosta significativamente dalla distribuzione dei valori.
# Risultano essere elementi estremi che potrebbero indurre il modello a errori di apprendimento, compromettendo la sua stabilità.
#
# Quindi, pur trattandosi di dati reali e corretti dal punto di vista clinico, si è scelto di rimuovere questi due specifici record dal dataset prima della fase di modellazione.

# %%
ds.drop(ds[ds['cell_diameter_um'] > 20].index, inplace=True)

ds.drop(ds[ds['platelet_count_per_ul'] > 540000].index, inplace=True)

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
# Per la maggior parte delle variabili si ha un distribuzione dei valori che somiglia a una curva normale, fatta eccezione per `cell_diameter_um nuclues_area_pct`. Realizziamo dei boxplot per categoria della variabile target per studiarne la dispersione

# %%
plt.figure(figsize=(15, 8))

sns.boxplot(
    data=ds, 
    x='disease_category', 
    y='cell_diameter_um', 
    palette='Set3',
    hue='disease_category',
    legend=False
)

plt.title('Analisi della Dispersione per Cell Diameter', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.show()

# %%
plt.figure(figsize=(15, 8))

sns.boxplot(
    data=ds, 
    x='disease_category', 
    y='nucleus_area_pct', 
    palette='Set3',
    hue='disease_category',
    legend=False
)

plt.title('Analisi della Dispersione per Nucleus Area %', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Per entrambe le feature si nota una separazione tra i vari IQR per classe, seppure nel caso di `cell diameter` si abbia comunque un po' di sovrapposizione. Nel caso di nucleus area invece si nota bene, la separazione fra alcune classi come _Normal_RBC_,_Normal_Platelet_,_Anemia_,_Sickle_Cell_Anemia_, che non hanno il nucleo, e _Leukemia_

# %% [markdown]
# Si procede a realizzare gli istogrammi anche per le altre feature numeriche come punteggi legati a caratteristiche morfologiche della cellula

# %%
scores = [
    ('chromatin_density', 'score'),
    ('cytoplasm_ratio', 'ratio'),
    ('circularity', 'score'),
    ('eccentricity', 'score'),
    ('granularity_score', 'score'),
    ('lobularity_score', 'score'),
    ('membrane_smoothness', 'score')
]

plt.figure(figsize=(25, 20)) 

for i, (col, label) in enumerate(scores,1):
    plt.subplot(3, 3, i)
    plt.title(col)
    plt.hist(ds[col])
    plt.xlabel(label)

plt.tight_layout()
plt.show()

# %% [markdown]
# Gli istogrammi rivelano la presenza di cluster e forti frequenze in alcuni range di valori. Di seguito un'analisi più approfondita mediante boxplot per classe

# %%
fig, axes = plt.subplots(3, 3, figsize=(25, 20))
axes = axes.flatten()

for i, (col, unit) in enumerate(scores):
    sns.boxplot(
        data=ds,
        x='disease_category',
        y=col,
        palette='Set3',
        hue='disease_category',
        legend=False,
        ax=axes[i]
    )
    
    axes[i].set_title(col, fontsize=14)
    axes[i].set_ylabel(unit)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# %% [markdown]
# Si nota generalmente, dai boxplot delle feature, che in base alla classe la distribuzione e dispersione dei valori varia.
#
# In particolare:
#
# *analisi dei singoli boxplot?*

# %% [markdown]
# ### Analisi feature legate all'immagine

# %% [markdown]
# Le feature `cell_area_px perimeter_px mean_r mean_g mean_b microscope_model magnification_x image_resolution_px` sono tutte legate all'immagine della cellula (l'immagine non è presente nel dataset).
#
# In particolare `image_resolution_px` è la dimensione dell'immagine su un asse cartesiano

# %% [markdown]
# Si creano gli istogrammi delle feature numeriche legate all'immagine per verificare la presenza di cluster

# %%
plt.figure(figsize=(25, 20)) 

plt.subplot(3, 3, 1)
plt.title('Cell area')
plt.hist(ds['cell_area_px'])
plt.xlabel('px')


plt.subplot(3, 3, 2)
plt.title('Cell perimeter')
plt.hist(ds['perimeter_px'])
plt.xlabel('px')


plt.subplot(3, 3, 3)
plt.title('Mean red')
plt.hist(ds['mean_r'])
plt.xlabel('px')


plt.subplot(3, 3, 4)
plt.title('Mean blue')
plt.hist(ds['mean_b'])
plt.xlabel('px')


plt.subplot(3, 3, 5)
plt.title('Mean green')
plt.hist(ds['mean_g'])
plt.xlabel('px')

plt.tight_layout()
plt.show()

# %% [markdown]
# Le dimensioni in pixel di componenti dell'immagine potrebbe dipendere dagli altri parametri relativi all'acquisizione dell'immagine quali `microscope_model magnification_x image_resolution_px` .
# Creando un grafico boxplot per categorie si possono notare differenze nella distribuzione dei valori

# %%
plt.figure(figsize=(15, 10))

plt.subplot(3,3,1)

sns.boxplot(
    data=ds, 
    x='magnification_x', 
    y='cell_area_px', 
    palette='Set3',
    hue='magnification_x',
    legend = False
)

plt.title('Cell area per magnification')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)

plt.subplot(3,3,2)

sns.boxplot(
    data=ds, 
    x='image_resolution_px', 
    y='cell_area_px', 
    palette='Set3',
    hue='image_resolution_px',
    legend = False
)

plt.title('Cell area per image resolution')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)

plt.subplot(3,3,3)

sns.boxplot(
    data=ds, 
    x='microscope_model', 
    y='cell_area_px', 
    palette='Set3',
    hue='microscope_model',
    legend = False
)

plt.title('Cell area per Microscope model')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Dai grafici si apprende che le caratteristiche relative all'acquisizione non hanno incidenza sulla distribuzione dei dati e che quindi `cell_area_px` non dipenda da queste variabili. Questo può essere dovuto al fatto che la feature `cell_area_px` fosse già stata normalizzata nel dataset o che siano state applicate altre forme di scaling o preprocessing dei dati

# %% [markdown]
# Le cellule malate potrebbero differire tra di loro per dimensione. Realizziamo un boxplot
# diviso per classe `disease_category` per vedere se la dispersione dei dati varia in base
# alla classe

# %%
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=ds, 
    x='disease_category', 
    y='cell_area_px', 
    palette='Set3',
    hue='disease_category'
)
plt.title('Cell area per disease category', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Notiamo che in base alla classe si ha una notevole differenza nella distribuzione dei valori. Le classi differiscono fra loro nelle mediane e dimensioni dei range interquantili. 
#
# Questo ci suggerisce che la feature potrebbe essere informativa per predire la variabile target

# %% [markdown]
# Si suppone inoltre che ci sia una correlazione tra `perimeter_px` e `cell_area_px`. Calcoliamo l'indice di correlazione

# %%
correlation_df = ds[['cell_area_px' ,'perimeter_px']].corr()
print(correlation_df.iloc[1,0])

# %% [markdown]
# Si nota una forte correlazione, quindi si può presupporre che anche `perimeter_px` possa essere molto determinante
# nella predizione. 
#
# Si realizza uno scatter plot per individuare meglio la presenza di cluster

# %%
disease_color_map = {"Leukemia" : "red",
                    "Anemia" : "orange",
                    "Sickle_Cell_Anemia" : "yellow",
                    "Infection" : "green",
                    "Artefact" : "black"}

# %%
sample = ds.sample(1000) 

fig, ax = plt.subplots(figsize=(8, 6))

for category, group in sample.groupby("disease_category"):
    color = disease_color_map.get(category, "blue")
    
    group.plot.scatter(
        x="cell_area_px", 
        y="perimeter_px", 
        c=color, 
        label=category,
        ax=ax
    )

ax.legend(title="Disease Category")

# %% [markdown]
# Si nota la formazione di cluster per alcune categorie di malattie. Si presuppone quindi che queste feature possano essere determinanti per la predizione della variabile target

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Collinearità e relazione tra variabili

# %% [markdown]
# Calcoliamo gli indici di collinearità fra le feature numeriche del dataset per comprendere quali feature sono correlate e dunque hanno una relazione di dipendenza le une dalle altre

# %%
numerical_features = [
    'cell_diameter_um',
    'nucleus_area_pct',
    'chromatin_density',
    'cytoplasm_ratio',
    'circularity',
    'eccentricity',
    'granularity_score',
    'lobularity_score',
    'membrane_smoothness',
    'cell_area_px',
    'perimeter_px',
    'mean_r',
    'mean_g',
    'mean_b',
    'stain_intensity',
    'wbc_count_per_ul',
    'rbc_count_millions_per_ul',
    'hemoglobin_g_dl',
    'platelet_count_per_ul',
    'mcv_fl',
    'mchc_g_dl',
    'magnification_x',
    'image_resolution_px']

correlations = ds[numerical_features].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(correlations, dtype=np.bool)    
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(correlations, mask=mask, cmap=cmap, center=0,annot = True, square=True, linewidths=.5, cbar_kws={"shrink": .5});

# %% [markdown]
# Dalla tabella notiamo come tutte le feature al di sotto della riga `stain_intensity` 
# abbiano poca correlazione con il resto del dataset. Già dagli istogrammi si poteva osservare come queste feature assumessero una distribuzione alquanto più simile a una gaussiana e che quindi probabilmente tali variabili assumessero una distribuzione normale e dunque indipendente da altri fattori

# %% [markdown]
# Sempre guardando la matrice di correlazione si notano le seguenti relazioni:
#
# * `chromatin_density` e `nucleus_area_pct` : Questo risultato sembra sottolineare un possibile errore nei dati. Infatti, le due feature dovrebbero essere legate da una collinearità inversa, perchè più piccolo il nucleo maggiore dovrebbe essere la densità della cromatina (che è contenuta solo nel nucleo della cellula). È necessario fare ulteriori investigazioni per comprendere il significato di questo risultato, si sottolinea anche che `nucleus_area_pct` non è la dimensione assoluta del nucleo, ma è un dato che dipende dalla dimensione della cellula
# * `cytoplasm_ratio` e `nucleus_area_pct` : le due feature sono inversamente correlate, era un risultato atteso in quanto il citoplasma occupa la parte di cellula non contenuta nel nucleo. Maggiore la dimensione del nucleo, minore lo spazio rimanente
# * `cytoplasm_ratio` vs `chromatin_density` : Sempre a seguito delle osservazioni fatte su `chromatin_density` e `nucleus_area_pct` e `cytoplasm_ratio` e `nucleus_area_pct` si deduce come in realtà questo rislutato possa apparire come apparentemente errato
# * `eccentricty` e `circularity` : i due punteggi hanno semanticamente significati opposti, è quindi atteso che ci fosse una collinearità inversa
# * `cell_area_px` e `cell_diameter_um` : L'aumentare della misura del diametro è perfettamente in linea con una maggiore dimensione in pixel della cellula nella foto, si rammenta che i valori in pixel sono già stati normalizzati e non dipendono dall'immagine  
# * `perimeter_px` e `cell_diameter_um` : Discorso analogo all'area della cellula
# * `perimeter_px` e `cell_area_px`: Come mostrato in precedenza, cellule più grandi in termini di area corrispondono a un perimetro maggiore
# * `mean_b`e `nucleus_area_pct`: Le immagini del dataset sono state acquisite a seguito del procedimento _Wright-Giemsa_ di macchiatura. Questa tecnica risalta i cromosomi nell'immagine colorandoli di blu. Un nucleo più grande implica che i cromosomi occupino più spazio nella foto, di conseguenza il colore blu sarà maggiormente predominante e questo porta ad alzare la media del valore del canale blu nella foto
# * `mean_b` e `cytoplasm_ratio` : a partire dalla relazione fra `mean_b`e `nucleus_area_pct` e `cytoplasm_ratio` e `nucleus_area_pct` si può comprendere il perchè queste feature siano inversamente collineari

# %% [markdown]
# Si suppone che le feature `chromatin_density` e `nucleus_area_pct` abbiano un valore di collinearità così elevato perchè si ha una forte concentrazione di istanze attorno al valore 0, in entrambe le feature. Queste istanze sono date principalmente da piastrine (platelets) e globuli rossi che non hanno un nucleo. Analogalmente non avendo un nucleo queste cellule avranno anche un valore di cromatina vicino allo 0

# %%
ds[(ds.nucleus_area_pct == 0)][['cell_type','disease_category','nucleus_area_pct','chromatin_density']]

# %% [markdown]
# Si prova a fare uno studio della collinearità più approfondito escludendo queste cellule

# %%
cells_wit_nucleus = ds[(ds.chromatin_density != 0) & (ds.nucleus_area_pct != 0)]
cells_wit_nucleus.plot.scatter("chromatin_density", "nucleus_area_pct")
cells_wit_nucleus[['chromatin_density','nucleus_area_pct']].corr()

# %% [markdown]
# Si nota che il valore di correlazione è calato drasticamente, guardando il nuovo valore si potrebbe dire che le feature sono debolmente correlate.
#
# Andando più a fondo si evidenzia come nucleus_area_pct, per definizione, non è una misura assoluta della dimensione del nucleo bensì relativa aslla dimensione totale della cellula. Invece la densità della cromatina dovrebbe dipendere dalla dimensione del nucleo in termini assoluti. Non è quindi informativo svolgere un confronto diretto tra le due variabili. Ad esempio, alcune cellule potrebbero avere un nucleo molto grande rapportato al volume totale (`nucleus_area_pct` alto) pur rimanendo cellule "piccole" con un nucleo "piccolo" e quindi avere valori di cromatina alti.
#
# Si prova a stimare una misura assoluta delle dimensioni del nucleo, creando una feature calcolata, e si esegue un'ulteriore analisi.

# %%
nucleus_area_px = ((cells_wit_nucleus.nucleus_area_pct/100) * cells_wit_nucleus.cell_area_px)

# %%
cells_wit_nucleus.insert(1,'nucleus_area_px',nucleus_area_px) 

# %%
cells_wit_nucleus.plot.scatter("chromatin_density", "nucleus_area_px")
cells_wit_nucleus[['chromatin_density','nucleus_area_px']].corr()

# %% [markdown]
# Le due variabili risultano essere moderatamente correlate negativamente, e questo risultato è in linea con il comportamento che la scienza ci dice dovrebbe verificarsi. 
#
# Da questo si deduce che la forte correlazione che avevamo osservato in origine è in realtà un artefatto dovuto alla distribuzione dei dati molto sbilanciata verso lo 0 per le due feature, dovuta a quelle tipologie di cellule che non hanno un nucleo. 
#
# Si procede a mantenere il dataset nello stato corrente verrà poi osservato nella fase di allenamento l'incidenza delle feature ed eventualmente si progetteranno soluzioni che considerano questo problema con il fine di migliorare l'accuratezza dei modelli.
#
# Eventualmente, si potrebbe considerare l'introduzione della variabile calcolata `nucleus_area_px`

# %% [markdown]
# # Preparazione dei dati

# %% [markdown]
# Si preparano ora i dati per essere elaborati dal modello.
#
# Si eliminano le features _anomaly_label_ e _cell_type_ poichè porterebbero ad un'apprendimento diretto. L'obiettivo del nostro modello è predire la _disease_category_ date le informazioni strettamente legate alla analisi della cellula, e non predire la malattia conoscendo giè il tipo di cellula o se l'elemento stesso è anomalo.
#
# Si procede inoltre a isolare anche la variabile target _disease_cateogry_ da predire.

# %%
target = ds['disease_category']
ds.drop(columns=['disease_category', 'anomaly_label', 'cell_type'], inplace=True)

# %%
to_encode = ['patient_age_group',
                'patient_sex',
                'staining_protocol',
                 'microscope_model',
                 'magnification_x',
                 'image_resolution_px',]

ds = pd.get_dummies(ds, columns=to_encode, dtype=int)

# %% [markdown]
# Essendo presenti features categoriche si procede a processare i dati attraverso il One-Hot Encoding

# %%
target.info()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ds, target, test_size=0.3, random_state=43, stratify=target)

# %%
y_train.head()

# %%
X_train.head()

# %% [markdown]
# # Addestramento e validazione

# %% [markdown]
# Si importano le librerie necessarie.

# %%
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
# %%
def plot_confusion_matrix(matrix, labels = None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,      # show counts
        fmt='d',         # integer format
        cmap='Blues',
        cbar_kws={"shrink": .5},
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# %%
def dump_statistics(model,X_train,X_test,y_train,y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    print('Classification report on train set \n' + classification_report(y_train,y_train_pred))
    print('Classification report on test set \n' + classification_report(y_test,y_test_pred))
    print('F1 weighted score on test {:.4f}'.format(metrics.f1_score(y_test, y_test_pred, average='weighted')))

# %% [markdown]
# ## Perceptron

# %% [markdown]
# Si allena un semplice perceptron e si analizza il comportamento, siccome dall'analisi si è notato che il dataset ha un forte sbilanciamento fra classi usiamo l'impostazione _balanced_ per il perceptron che regola l'aggiustamento dei pesi in base al bilanciamento delle classi

# %%
std_perceptron = Pipeline([
    ('std', StandardScaler()),
    ('perceptron', Perceptron(n_jobs=-2, early_stopping=True,class_weight='balanced'))
])

parameters = {
    'std': [None, StandardScaler()],
    'perceptron__penalty': [None, 'l1', 'l2', 'elasticnet'],
    'perceptron__alpha': [0.0001, 0.001, 0.01, 1],
    'perceptron__tol': [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6],
}

perceptron_search = GridSearchCV(std_perceptron, parameters, n_jobs=-2, scoring='f1_weighted')

# %%
perceptron_search.fit(X_train, y_train)

# %%
print('Best parameters:', perceptron_search.best_params_)

# %% [markdown]
# Notiamo in particolare che la grid search ha selezionato una versione che attua normalizzazione sulle feature, il che è comprensibile viste le disparate scale e range di valori, e una regolarizzazione l1.
#
# Di seguito le variabili che sono state azzerate, notiamo alcune variabili numeriche e alcune variabili binarie associate a precedenti variabili categoriche per cui era stato eseguito il one hot encoding. Principalmente sono state azzerate variabili per cui si era già osservata la mancata presenza di cluster e per cui non c'erano forti separazioni negli IQR suddivisi per classi

# %%
coeff = pd.Series(perceptron_search.best_estimator_[1].coef_[0])
coeff[coeff == 0]

# %% [markdown]
# Si calcola accuratezza e f1 score del modello, notiamo già un risultato promettente per essere il primo allenamento

# %%
dump_statistics(perceptron_search,X_train,X_test,y_train,y_test)

# %%
cm = confusion_matrix(y_test, perceptron_search.predict(X_test))
plot_confusion_matrix(cm, perceptron_search.classes_)

# %% [markdown]
# Dalla matrice di confusione si nota che il modello fatica a distinguere fra le classi Anemia, Normal_RBC e Sickle_cell_anemia. Questo è comprensibile in quanto sono tutte condizioni di malattia legate ai globuli rossi (Normal_RBC). 
#
# Analogalmente si nota una difficoltà nella classificazione di Infection e Normal_WBC, in particolare il modello è poco preciso nel classificare le cellule di classe infezione come tali. Ciò può essere legato a motivazioni analoghe a Normal_RBC e alla mancanza di bilanciamento delle classi

# %% [markdown]
# Di seguito le feature evidenziate come più importanti dall'apprendimento

# %%
np.abs(coeff).nlargest(4).plot(kind='barh')

# %% [markdown]
# Si fa un secondo tentativo con l'espansione polinomiale

# %%
poly_perceptron = Pipeline([
    ('std', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('perceptron', Perceptron(n_jobs=-2, early_stopping=True,class_weight='balanced'))
])

parameters = {
    'std': [None, StandardScaler()],
    'perceptron__penalty': [None, 'l1', 'l2', 'elasticnet'],
    'perceptron__alpha': [0.0001, 0.001, 0.01, 1],
    'perceptron__tol': [1e-9, 1e-6, 1e-3, 1, 1e3, 1e6],
}

poly_perceptron_search = GridSearchCV(poly_perceptron, parameters, n_jobs=-2, scoring='f1_weighted')

# %%
poly_perceptron_search.fit(X_train,y_train)

# %%
dump_statistics(poly_perceptron_search,X_train,X_test,y_train,y_test)

# %%
print(poly_perceptron_search.best_params_)

# %%
perceptron_search.best_params_

# %% [markdown]
# Si nota che l'espansione polinomiale ha portato a risultati migliori, guardando in particolare gli f1-score. Usare feature polinomiali che introducano non linearità sembra quindi essere utile per modellare al meglio il dataset.
#
# È stato eseguito un altro tentativo provando ad allenare un perceptron semplice rimuovendo le variabili azzerate dalla L1 e ottenuto un f1-weighted score di 0.7981 che è quindi peggiore. Alterare lo spazio delle feature ha portato quindi a peggiori performance. Era stato eseguito anche un altro esperimento aumentando a 3 il grado del polinomio di `poly_perceptron` portando a risultati migliori, si riconferma quindi che l'espansione polinomiale aiuta a modellare meglio la separazione fra classi
# %% [markdown]
# ### Logistic Regression

# %% [markdown]
# Il modello permette di ottenere un piano di separazione lineare tra le classi
#
# Le scelte principali effettuate sono state:
#
# - standardizzazione dei dati tramite StandardScaler;
# - utilizzo della Logistic Regression come classificatore lineare;
# - scelta del solver saga;
# - impostazione di un numero massimo di iterazioni pari a 5000;
# - fissazione del random_state per garantire la riproducibilità dei risultati.
#
# Mentre la ricerca ha considerato:
#
# - metodo di penalizzazione/regolarizzazione;
# - valore di C, inverso dell’intensità di regolarizzazione;
# - valore di tolleranza tol per il criterio di arresto.

# %%
std_lr = Pipeline([
    ('std', StandardScaler()),
    ('lr', LogisticRegression(
        solver='saga',
        max_iter=5000,
        random_state=42
    ))
])

parameters = {
    'lr__penalty': ['l1', 'l2'],
    'lr__C': [0.1, 3, 10, 50, 100],
    'lr__tol': [1e-2,1e-3, 1e-2]
}

lr_search = GridSearchCV(std_lr, parameters, cv=5, n_jobs=-2, return_train_score=True, scoring='f1_weighted')

# %%
lr_search.fit(X_train, y_train)

# %%
print('Best parameters:', lr_search.best_params_)

# %%
lr_coeff = pd.Series(lr_search.best_estimator_[1].coef_[0], index=X_train.columns)

# %% [markdown]
# Di seguito le features considerate le più importanti dall'apprendimento:

# %%
np.abs(lr_coeff).nlargest(4).plot(kind='barh')

# %% [markdown]
# E' calcolata l'accuratezza e lo score del modello.

# %%
dump_statistics(lr_search,X_train,X_test,y_train,y_test)

# %%
lr_cm = confusion_matrix(y_test, lr_search.predict(X_test))
plot_confusion_matrix(lr_cm, lr_search.classes_)

# %% [markdown]
# Dalla matrice di confusione si nota che il modello ha difficoltà a classificare cellule di tipo _infection_ piuttosto che _normal_WBC_ e viceversa, così come _anemia_ e _normal_RBC_ ma si ha una maggiore accuratezza rispetto ai modelli precedenti.

# %% [markdown]
# Eseguire un'espansione polinomiale si è rivelato utile per il perceptron. Si è provato a svolgere lo stesso tentativo ma usare un'espansione polinomiale con la regressione logistica era troppo complesso in termine di costo computazionale. Si è pensato quindi di usare Nystroem per fornire un'approssimazione di funzioni kernel e quindi portare le variabili in uno spazio a maggiore dimensionalità senza calcolare le nuove variabili

# %%
from sklearn.kernel_approximation import Nystroem

kernel_lr = Pipeline([
    ('std', StandardScaler()),
    ('kernel', Nystroem(random_state=42)),
    ('kernel_std', StandardScaler()),
    ('lr', LogisticRegression(
        solver='saga',
        max_iter=5000,
        random_state=42
    ))
])

parameters = {
    'kernel__kernel': ['rbf', 'poly', 'sigmoid'],
    # n_components is the number of features to construct (higher = better approximation but slower)
    'kernel__n_components': [50, 100, 200],
    'lr__penalty': ['l1', 'l2'],
    'lr__C': [0.1, 3, 10],
    'lr__tol': [1e-3, 1e-2]
}

kernel_lr_search = GridSearchCV(kernel_lr, parameters, cv=5, n_jobs=-2, return_train_score=True, scoring='f1_weighted')

# %%
kernel_lr_search.fit(X_train, y_train)

# %%
print('Best parameters:', kernel_lr_search.best_params_)

# %%
dump_statistics(kernel_lr_search,X_train,X_test,y_train,y_test)

# %% [markdown]
# In questo caso si nota un minimo miglioramento. Si ricorda che in questo caso si stanno usando approssimazioni di funzioni kernel e che questo porta intrisecamente ad imprecisione. Inoltre algoritmi come SVM, proprio per natura progettuale, funzionano meglio con le funzioni kernel. Infatti, con SVM si ha la possibilità di usare direttamente il kernel trick perchè la dimensione della matrice del kernel è molto più contenuta rispetto a logistic regression. Permettendo quindi di portare le feature in uno spazio ad alta dimensionalità senza errori di approssimazione

# %% [markdown]
# ### SVM

# %%
std_svm = Pipeline([
    ('std', StandardScaler()),
    ('svm', SVC())
])

parameters = {
    'svm__kernel': ['rbf', 'linear', 'poly'],
    'svm__C': [0.01, 0.1, 1, 10, 50, 100],
}

svm_search = GridSearchCV(std_svm, parameters, cv=3, n_jobs=-2, return_train_score=True, scoring='f1_weighted')

# %%
svm_search.fit(X_train, y_train)

# %%
print('Best parameters:', svm_search.best_params_)  

# %%
svm_coeff = pd.Series(svm_search.best_estimator_[1].support_vectors_[0], index=X_train.columns)

# %% [markdown]
# Di seguito si evidenziano le features più rilevanti per il modello

# %%
np.abs(svm_coeff).nlargest(4).plot(kind='barh')

# %%
dump_statistics(svm_search,X_train,X_test,y_train,y_test)

# %%
svm_cm = confusion_matrix(y_test, svm_search.predict(X_test))
plot_confusion_matrix(svm_cm, svm_search.classes_)

# %% [markdown]
# Dalla matrice si evidenzia, come per gli altri modelli, la difficoltà nel distinguere le cellule di categoria _infection_ e _anemia_ ma si nota una maggiore precisione nella classificazione di queste.

# %% [markdown]
# Il dataset presenta molti casi di collinearità che spesso portano ad
# un impoverimento delle performance specialmente in modelli lineari.
# Si fa un tenativo per rimuovere le feature con forte collinearità e
# vedere se l'accuratezza migliora

# %%
X_train_no_corr = X_train.drop(columns=['cytoplasm_ratio','circularity','cell_area_px','perimeter_px'])
X_test_no_corr = X_test.drop(columns=['cytoplasm_ratio','circularity','cell_area_px','perimeter_px'])

# %%
svm_search_no_corr = GridSearchCV(std_svm, parameters, cv=3, n_jobs=-2, return_train_score=True, scoring='f1_weighted')

# %%
svm_search_no_corr.fit(X_train_no_corr,y_train)

# %%
dump_statistics(svm_search_no_corr,X_train_no_corr,X_test_no_corr,y_train,y_test)

# %% [markdown]
# L'accuratezza è peggiorata, questo signica che le variabili eliminate, seppur presentano correlazione. Contengono informazioni importanti per l'apprendimento della separazione tra classi

# %% [markdown]
# ### Classification Tree

# %%
from sklearn.tree import DecisionTreeClassifier

std_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('tree',DecisionTreeClassifier(random_state=42))  
])
parameters = {
    "tree__max_depth": range(4, 10),
    "tree__min_samples_split": [.05, .1, .15],
}
gs_tree = GridSearchCV(std_tree, parameters, cv=3, n_jobs = -2, return_train_score= True, scoring='f1_weighted')
gs_tree.fit(X_train, y_train)

# %%
dump_statistics(gs_tree,X_train,X_test,y_train,y_test)

# %%
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 6))
plot_tree(gs_tree.best_estimator_[1],feature_names=X_train.columns, max_depth=3, filled=True, fontsize=8);

# %% [markdown]
# ### Random Forest

# %%
import math
std_forest = Pipeline([
    ("scaler", StandardScaler()),
    ("forest", RandomForestClassifier(n_jobs=-1, random_state=42))
])


parameters = {
        "scaler": [None, StandardScaler()],
        "forest__n_estimators": [100, 200, 500],
        "forest__max_depth": [None, 5, 10, 20],
        "forest__min_samples_split": [2, 5, 10],
        "forest__max_features": ["sqrt", "log2", None],
        "forest__min_samples_leaf": [1, 2, 5]
}
forest_search = GridSearchCV(std_forest, parameters, cv=3, n_jobs=-2, return_train_score= True, scoring='f1_weighted')#balanced_accuracy

# %%
forest_search.fit(X_train, y_train)

# %%
print('Best parameters:', forest_search.best_params_)  

# %%
dump_statistics(forest_search,X_train,X_test,y_train,y_test)

# %%
forest_cm = confusion_matrix(y_test, forest_search.predict(X_test))
plot_confusion_matrix(forest_cm, forest_search.classes_)

# %%
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 6))
plot_tree(forest_search.best_estimator_[1].estimators_[0],feature_names=X_train.columns, max_depth=3, filled=True, fontsize=8);

# %% [markdown]
# ### XGBoost

# %%
# !pip install xgboost

# %%
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# %%
std_xgb = Pipeline([
    ('std', StandardScaler()),
    ('xgb', XGBClassifier(n_jobs=8, objective="multi:softprob"))
])

parameters = {
    'xgb__eta': [0.01, 0.05, 0.1],
    'xgb__min_child_weight': [4, 10],
    'xgb__max_depth': [3, 4, 5, 6],
    'xgb__n_estimators': [150, 300, 500],
    'xgb__alpha': [0.0001, 0.001, 0.01]
}

xgb_search = GridSearchCV(std_xgb, parameters, cv=3, n_jobs=-2, return_train_score=True, scoring='f1_weighted')

# %%
xgb_search.fit(X_train, y_train_enc)

# %%
print('Best parameters:', xgb_search.best_params_)  

# %%
xgb_coeff = pd.Series(xgb_search.best_estimator_[1].feature_importances_[0], index=X_train.columns)

# %%
np.abs(xgb_coeff).nlargest(4).plot(kind='barh')

# %%
dump_statistics(xgb_search,X_train,X_test,y_train_enc,y_test_enc)

# %%
xgb_cm = confusion_matrix(y_test_enc, xgb_search.predict(X_test))
plot_confusion_matrix(xgb_cm, xgb_search.classes_)

# %% [markdown]
# # Ottimizzazione

# %% [markdown]
# ## Bilanciamento delle classi

# %% [markdown]
# Si prova a fare un tentativo di bilanciamento delle classi per vedere se questo migliora le performance del modello XGBoost.
# Usiamo SMOTE per generare dati sintetici su cui allenare il modello

# %%
from imblearn.over_sampling import SMOTE

# %%
balancer = SMOTE(random_state=42)
X_res,y_res = balancer.fit_resample(X_train,y_train)

# %%
X_res["disease_category"] = y_res
counts = X_res['disease_category'].value_counts()
num_categorie = len(counts)

colori = plt.cm.tab10(range(num_categorie)) 

plt.bar(counts.index, counts.values, color=colori)
plt.title('Disease category')
plt.xlabel('Categorie')
plt.ylabel('Conteggio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X_res.drop(columns=['disease_category'], inplace=True)

# %%
y_train_res_enc = le.transform(y_res)
xgb_search.best_estimator_.fit(X_res,y_train_res_enc)

# %%
dump_statistics(xgb_search.best_estimator_,X_res,X_test,y_train_res_enc,y_test_enc)

# %% [markdown]
# Notiamo rispetto al modello principale un lieve peggioramento generale. In particolare il modello è meno preciso quindi, aumentano i casi di falsi positivi. Questo risultato può essere dovuto a una situazione di overfitting, infatti si nota un'estrema precisione nel training set. L'introduzione di un elevato numero di campioni sintetici, specialmente per classi poco rappresentate, può aver portato il modello a peggiorare la sua capacità di generalizzazione e ad appredere informazioni che non corrispondono totalmente con la realtà
