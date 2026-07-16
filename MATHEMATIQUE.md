# Théorie Mathématique et Algorithmes
## Régression Logistique et Classification One-vs-All

---

## Table des matières

1. [Régression Logistique](#régression-logistique)
2. [Fonction de Coût](#fonction-de-coût)
3. [Gradient Descent](#gradient-descent)
4. [Classification One-vs-All](#classification-one-vs-all)
5. [Normalisation des Données](#normalisation-des-données)
6. [Algorithmes Implémentés](#algorithmes-implémentés)
7. [Optimisations et Considérations Numériques](#optimisations-et-considérations-numériques)

---

## Régression Logistique

### Concept Fondamental

La régression logistique est un modèle de **classification linéaire** qui prédit la probabilité qu'une instance appartienne à une classe donnée.

Contrairement à la régression linéaire qui prédit une valeur continue, la régression logistique utilise une fonction d'activation non-linéaire pour mapper les prédictions à l'intervalle [0, 1].

### Fonction Logistique (Sigmoid)

La fonction sigmoid transforme toute valeur réelle en probabilité entre 0 et 1 :

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Où $z = \theta^T x$ (produit scalaire du vecteur de poids et de features).

**Propriétés** :
- $\sigma(0) = 0.5$
- $\lim_{z \to \infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$

### Hypothèse du Modèle

Pour une instance $x_i$, la prédiction est :

$$h_\theta(x_i) = \sigma(\theta^T x_i) = \frac{1}{1 + e^{-\theta^T x_i}}$$

Cette valeur représente la probabilité prédite que $y_i = 1$ (l'instance appartient à la classe positive).

---

## Fonction de Coût

### Régression Linéaire vs Logistique

En régression linéaire, on utilise l'erreur quadratique moyenne (MSE). Cependant, avec la sigmoid, cette fonction serait **non-convexe** et aurait plusieurs minima locaux, rendant l'optimisation difficile.

Pour la régression logistique, on utilise **l'entropie croisée binaire** (Binary Cross-Entropy), qui est convexe.

### Entropie Croisée Binaire

Pour un seul exemple $(x_i, y_i)$ où $y_i \in \{0, 1\}$ :

$$\text{Cost}(h_\theta(x_i), y_i) = -\left[ y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i)) \right]$$

**Interprétation** :
- Si $y_i = 1$ : $\text{Cost} = -\log(h_\theta(x_i))$
  - Si $h_\theta(x_i) \approx 1$ (bonne prédiction) : Cost $\approx 0$
  - Si $h_\theta(x_i) \approx 0$ (mauvaise prédiction) : Cost $\approx \infty$

- Si $y_i = 0$ : $\text{Cost} = -\log(1-h_\theta(x_i))$
  - Si $h_\theta(x_i) \approx 0$ (bonne prédiction) : Cost $\approx 0$
  - Si $h_\theta(x_i) \approx 1$ (mauvaise prédiction) : Cost $\approx \infty$

### Fonction de Coût Globale

Pour $m$ exemples d'entraînement :

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \text{Cost}(h_\theta(x_i), y_i)$$

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_\theta(x_i)) + (1-y_i) \log(1-h_\theta(x_i)) \right]$$

**Objectif** : Minimiser $J(\theta)$ par rapport aux paramètres $\theta$.

---

## Gradient Descent

### Concept Fondamental

Le gradient descent est un algorithme d'optimisation itératif qui cherche les paramètres minimisant la fonction de coût.

À chaque itération, on se déplace dans la direction du **gradient négatif** (direction de plus forte descente) avec un pas de taille proportionnel au **taux d'apprentissage** ($\alpha$).

### Calcul du Gradient

Le gradient de $J(\theta)$ par rapport à $\theta_j$ est :

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x_i) - y_i \right) x_{ij}$$

**En notation vectorielle** :

$$\nabla J(\theta) = \frac{1}{m} X^T \left( h_\theta(X) - y \right)$$

Où :
- $X$ est la matrice $m \times n$ (samples × features)
- $h_\theta(X)$ est le vecteur des prédictions
- $y$ est le vecteur des étiquettes

### Règle de Mise à Jour

À chaque itération :

$$\theta := \theta - \alpha \nabla J(\theta)$$

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

Où $\alpha$ est le **taux d'apprentissage** (généralement 0.01 à 0.1).

**Hyperparamètres importants** :
- $\alpha$ trop petit → Convergence lente
- $\alpha$ trop grand → Divergence possible
- Nombre d'itérations → Trade-off entre temps et convergence

---

## Classification One-vs-All

### Problème Multi-classe

Notre problème a **4 classes** (Maisons Poudlard) :
- Gryffindor
- Hufflepuff
- Ravenclaw
- Slytherin

La régression logistique binaire ne peut pas résoudre directement cela. On utilise la stratégie **One-vs-All** (One-vs-Rest).

### Stratégie One-vs-All

1. **Créer 4 classifieurs binaires**, un par classe
2. Pour chaque classifieur $k$ :
   - Étiquette positive : appartient à la classe $k$
   - Étiquette négative : n'appartient pas à la classe $k$
3. Entraîner chaque classifieur indépendamment

### Prédiction Multi-classe

Pour une nouvelle instance $x$, on calcule les 4 probabilités :

$$h_\theta^{(1)}(x), h_\theta^{(2)}(x), h_\theta^{(3)}(x), h_\theta^{(4)}(x)$$

La **classe prédite** est celle avec la probabilité maximale :

$$\text{Classe prédite} = \arg\max_k h_\theta^{(k)}(x)$$

### Exemple Concret

Entraînement du classifieur Gryffindor vs Rest :

| Étudiant | Astronomy | Charms | ... | Gryffindor (y) |
|----------|-----------|--------|-----|----------------|
| 1        | -0.51     | -0.23  | ... | 1 (Gryffindor) |
| 2        | 0.68      | 0.95   | ... | 0 (Autres)     |
| 3        | 0.12      | -0.67  | ... | 0 (Autres)     |
| ...      | ...       | ...    | ... | ...            |

Après entraînement, pour un nouvel étudiant :
- $h_{\theta}^{(Gryffindor)}(x) = 0.75$ (75% de chance Gryffindor)
- $h_{\theta}^{(Hufflepuff)}(x) = 0.20$
- $h_{\theta}^{(Ravenclaw)}(x) = 0.03$
- $h_{\theta}^{(Slytherin)}(x) = 0.02$

**Prédiction** : Gryffindor (probabilité maximale)

---

## Normalisation des Données

### Pourquoi Normaliser ?

Les features ont des unités et des échelles différentes :
- Astronomy : [-600, 700]
- Charms : [-300, 300]
- Ancient Runes : [100, 650]

Cela crée plusieurs problèmes :
1. **Convergence lente** — Features avec grande amplitude dominent le gradient
2. **Instabilité numérique** — Débordement/sous-débordement possibles
3. **Learning rate suboptimal** — Même $\alpha$ inadapté pour toutes les features

### Standardisation (Z-score Normalization)

Pour chaque feature $x_j$ :

$$x_j^{\text{norm}} = \frac{x_j - \mu_j}{\sigma_j}$$

Où :
- $\mu_j$ = moyenne de feature $j$
- $\sigma_j$ = écart-type de feature $j$

**Résultat** : Chaque feature normalisée a moyenne ≈ 0 et écart-type ≈ 1

### Données Manquantes

Les données d'entraînement contiennent des valeurs manquantes (NaN). Stratégie :

1. **Imputation par la moyenne** : Remplacer NaN par la moyenne de la feature calculée sur les données d'entraînement
2. **Appliquer la même transformation en prédiction** : Utiliser les statistiques du train, pas du test

**Code** :
```python
mean = dataset_train[feature].mean()  # Calculé une fois sur l'entraînement
test_data[feature] = test_data[feature].fillna(mean)
```

---

## Algorithmes Implémentés

### 1. Batch Gradient Descent (BGD)

**Mise à jour** :
$$\theta := \theta - \alpha \frac{1}{m} X^T(h_\theta(X) - y)$$

**Pseudocode** :
```
θ ← vecteur zéro
pour iter = 1 à num_iterations:
    h ← sigmoid(X @ θ)
    gradient ← (X.T @ (h - y)) / m
    θ ← θ - α * gradient
retourner θ
```

**Caractéristiques** :
- ✅ Convergence garantie vers minima global (fonction convexe)
- ✅ Trajectoire lisse et stable
- ✅ Prévisible et facile à tuner
- ❌ Lent sur très grands datasets
- ❌ Calcul complet à chaque itération

**Hyperparamètres** (implémentation) :
- Iterations : 10,000
- Learning rate : 0.1
- Batch size : $m$ (tous les samples)

---

### 2. Stochastic Gradient Descent (SGD)

**Mise à jour** :
$$\theta := \theta - \alpha (h_\theta(x_i) - y_i) x_i$$

Seulement avec l'exemple $i$ choisi aléatoirement.

**Pseudocode** :
```
θ ← vecteur zéro
pour epoch = 1 à num_epochs:
    pour chaque sample (x_i, y_i):
        i ← indice aléatoire
        h_i ← sigmoid(x_i @ θ)
        gradient ← (h_i - y_i) * x_i
        θ ← θ - α * gradient
retourner θ
```

**Caractéristiques** :
- ✅ Très rapide (une comparaison par itération)
- ✅ Échappe aux minima locaux grâce au bruit
- ✅ Excellent pour données streaming
- ❌ Convergence très bruitée (zigzag)
- ❌ Moins stable que BGD

**Hyperparamètres** (implémentation) :
- Epochs : 100
- Learning rate : 0.1
- Batch size : 1 (un sample aléatoire)

**Raison du bruit** :
Avec un seul sample, le gradient est **estimé**, pas exact. Cela introduit de la variance mais aussi de l'exploration.

---

### 3. Mini-batch Gradient Descent

**Mise à jour** :
$$\theta := \theta - \alpha \frac{1}{b} X_b^T(h_\theta(X_b) - y_b)$$

Où $X_b$ est un batch de $b$ samples (typiquement 16, 32, ou 64).

**Pseudocode** :
```
θ ← vecteur zéro
pour epoch = 1 à num_epochs:
    pour chaque batch de taille batch_size:
        X_batch, y_batch ← samples du batch
        h_batch ← sigmoid(X_batch @ θ)
        gradient ← (X_batch.T @ (h_batch - y_batch)) / batch_size
        θ ← θ - α * gradient
retourner θ
```

**Caractéristiques** :
- ✅ Meilleur compromis entre BGD et SGD
- ✅ Efficace computationnellement (vectorisé)
- ✅ Convergence plus stable que SGD
- ✅ Adapté aux GPU (mini-batches parallélisables)
- ⚠️ Hyperparamètre supplémentaire (batch size)

**Hyperparamètres** (implémentation) :
- Epochs : 100
- Learning rate : 0.1
- Batch size : 32

---

## Comparaison des Trois Algorithmes

### Tableau Récapitulatif

| Aspect | BGD | SGD | Mini-batch |
|--------|-----|-----|-----------|
| **Données par itération** | Tous ($m$) | 1 | $b$ (32) |
| **Itérations totales** | 10,000 | 100 epochs × $m$ | 100 epochs × ($m$/$b$) |
| **Vitesse par itération** | Lente | Très rapide | Rapide |
| **Stabilité** | ✅ Très stable | ❌ Bruitée | ✅ Stable |
| **Convergence** | ✅ Lisse | ⚠️ Zigzag | ✅ Lisse + rapide |
| **Minima locaux** | ⚠️ Risque | ✅ Échappe grâce au bruit | ✅ Équilibré |
| **Efficacité mémoire** | ❌ Pauvre | ✅ Excellente | ✅ Bonne |
| **Parallélisation** | ❌ Difficile | ❌ Impossible | ✅ Facile (GPU/CPU) |

### Trajectoires Schématiques

```
Fonction de coût au cours de l'entraînement :

BGD (lisse) :
J(θ)
  │
  │  \
  │   \___
  │       \
  │        \___
  └─────────────→ Itérations

SGD (bruitée) :
J(θ)
  │     ╱╲╱╲
  │    ╱  ╲  ╲___
  │   ╱    ╲     \
  │  ╱      ╲     ╲
  └─────────────→ Itérations

Mini-batch (équilibrée) :
J(θ)
  │  ╲
  │   ╲__
  │      ╲__
  │         ╲
  └─────────────→ Itérations
```

---

## Optimisations et Considérations Numériques

### 1. Débordement de la Fonction Sigmoid

**Problème** : Pour $z > 700$, $e^{-z} \to 0$ et on obtient une division par zéro.

**Implémentation robuste** :
```python
def sigmoid_stable(z):
    # Cliper z pour éviter le débordement
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

### 2. Terme de Biais

Les modèles incluent un terme de biais (intercept) $\theta_0$ constant.

**Implémentation** :
```python
# Ajouter une colonne de 1 au début de X
X_with_bias = np.column_stack([np.ones(m), X])
# θ[0] devient le terme de biais
```

### 3. Éviter Sous-débordement Logarithmique

**Problème** : $\log(h)$ où $h \to 0$ donne $-\infty$.

**Solution** :
```python
# Ajouter epsilon (petite constante)
cost = -np.mean(y * np.log(h + 1e-15) + 
                (1-y) * np.log(1-h + 1e-15))
```

### 4. Initialisation des Poids

**BGD** : Initialiser à zéro fonctionne bien (fonction convexe)
```python
theta = np.zeros(n_features)
```

**SGD/Mini-batch** : Peut être initialisé à zéro aussi, mais ajouter du bruit aide parfois

### 5. Convergence et Critères d'Arrêt

**BGD** : Nombre fixe d'itérations (simple, reproductible)

**SGD/Mini-batch** : Arrêter quand :
- Nombre d'epochs atteint
- Validation loss augmente (early stopping)
- Gradient norme < seuil

### 6. Régularisation (Bonus)

Ajouter une pénalité L2 pour éviter l'overfitting :

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [...] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Nouveau gradient :
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (...) + \frac{\lambda}{m} \theta_j$$

Effet : Réduit les poids et améliore la généralisation.

---

## Métriques d'Évaluation

### Accuracy (Précision Globale)

$$\text{Accuracy} = \frac{\text{Prédictions correctes}}{\text{Total prédictions}}$$

$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}(y_{\text{pred}}^{(i)} = y_{\text{true}}^{(i)})$$

**Interprétation** : Pourcentage de prédictions correctes (notre target : ≥ 98%)

### Matrice de Confusion (Multi-classe)

Exemple avec 4 classes (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) :

```
                 Prédictions
                 G   H   R   S
Vraies G  │  85  5   3   2
Vraies H  │  4   90  2   4
Vraies R  │  2   1   88  9
Vraies S  │  1   3   10  86
```

Diagonale = prédictions correctes

---

## Exemple Complet : Prédiction Pas à Pas

### Données d'Entrée

Nouvel étudiant avec features normalisées :
$$x = \begin{pmatrix} 1 \\ -0.35 \\ 0.82 \\ -0.12 \\ 0.56 \end{pmatrix}$$
(avec biais en première position)

Poids entraînés pour Gryffindor :
$$\theta^{(G)} = \begin{pmatrix} 0.15 \\ 0.42 \\ -0.81 \\ 0.33 \\ -0.25 \end{pmatrix}$$

### Calcul

1. **Produit scalaire** :
$$z = \theta^T x = 0.15(1) + 0.42(-0.35) + (-0.81)(0.82) + 0.33(-0.12) + (-0.25)(0.56)$$
$$z = 0.15 - 0.147 - 0.664 - 0.040 - 0.140 = -0.821$$

2. **Sigmoid** :
$$h = \sigma(-0.821) = \frac{1}{1 + e^{0.821}} = \frac{1}{1 + 2.273} = 0.305$$

**Interprétation** : 30.5% de chance que l'étudiant soit Gryffindor

3. **Comparaison avec autres classes** :
- Gryffindor : 0.305
- Hufflepuff : 0.450
- Ravenclaw : 0.180
- Slytherin : 0.065

**Classe prédite** : Hufflepuff (maximum)

---

## Ressources Additionnelles

### Livres et Papiers

- **"The Elements of Statistical Learning"** (Hastie, Tibshirani, Friedman)
- **"Deep Learning"** (Goodfellow, Bengio, Courville) — Chapitre 5
- **Andrew Ng's ML Course** — Logistic Regression lectures

### Implémentations Avancées

- Momentum et Nesterov Accelerated Gradient (NAG)
- Adam Optimizer
- Adaptive Learning Rates (AdaGrad, RMSprop)
- Batch Normalization

---

## Conclusion

La régression logistique est un algorithme **simple mais puissant** :
- ✅ Interprétable
- ✅ Efficace computationnellement
- ✅ Résout de vrais problèmes
- ✅ Fondation pour réseaux de neurones

Les trois variantes de gradient descent offrent différents compromis entre :
- **Stabilité vs Vitesse**
- **Convergence vs Exploration**
- **Mémoire vs Calcul**

Choisir en fonction de vos données, hardware, et objectifs d'optimisation.
