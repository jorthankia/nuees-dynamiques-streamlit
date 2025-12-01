import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from streamlit.components.v1 import html


# ============================================
#       Dynamic Clusters (Nuées dynamiques)
# ============================================

class DynamicClusters:
    def __init__(self, K, n_standards, max_iter=50, tol=1e-6, random_state=None):
        self.K = K
        self.ni = n_standards
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.history = []

    def distance_to_cluster(self, x, Ei):
        if len(Ei) == 0:
            return np.inf
        return np.mean(np.linalg.norm(Ei - x, axis=1))

    def R_value(self, x, i, L):
        d_i = self.distance_to_cluster(x, L[i])
        d_all = np.array([self.distance_to_cluster(x, L[j]) for j in range(self.K)])
        return d_i / np.sum(d_all)

    def fit(self, X):
        N = len(X)
        rng = np.random.default_rng(self.random_state)

        indices = rng.choice(N, self.K * self.ni, replace=False)
        L = []
        for i in range(self.K):
            L.append(X[indices[i*self.ni:(i+1)*self.ni]])

        for iteration in range(self.max_iter):
            self.history.append([Ei.copy() for Ei in L])

            D_matrix = np.zeros((N, self.K))
            for i in range(self.K):
                D_matrix[:, i] = cdist(X, L[i]).mean(axis=1)

            clusters = np.argmin(D_matrix, axis=1)

            new_L = []
            change = 0

            for i in range(self.K):
                Ci = X[clusters == i]

                if len(Ci) == 0:
                    new_L.append(L[i])
                    continue

                R_values = np.array([self.R_value(x, i, L) for x in Ci])
                idx = np.argsort(R_values)[: self.ni]
                Ei_new = Ci[idx]

                change += np.linalg.norm(Ei_new - L[i])
                new_L.append(Ei_new)

            if change < self.tol:
                break

            L = new_L

        self.L = L
        self.labels_ = clusters
        return self


# ============================================
#          Chargement Dataset IRIS
# ============================================

def load_data():
    data = load_iris()
    X = data.data
    X = StandardScaler().fit_transform(X)
    X_2D = PCA(n_components=2).fit_transform(X)
    return X_2D


# ============================================
#           Plot partition finale
# ============================================

def plot_results(X, model):
    labels = model.labels_
    K = model.K

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(K):
        ax.scatter(X[labels == i, 0], X[labels == i, 1], s=40, label=f"Classe {i+1}")
        ax.scatter(model.L[i][:, 0], model.L[i][:, 1], s=200, marker='X')

    ax.set_title("Nuées dynamiques — Partition finale")
    ax.legend()
    ax.grid()

    return fig


# ============================================
#         Animation évolutive (HTML)
# ============================================

def animate_clusters(X, model):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.close()

    def update(frame):
        ax.clear()
        L = model.history[frame]

        ax.scatter(X[:, 0], X[:, 1], s=10, color='gray', alpha=0.3)

        for i in range(model.K):
            ax.scatter(L[i][:, 0], L[i][:, 1], s=200, marker='X', label=f"Classe {i+1}")

        ax.set_title(f"Évolution des étalons — itération {frame}")
        ax.legend()
        ax.grid()

    anim = FuncAnimation(fig, update, frames=len(model.history), interval=900)
    return anim.to_jshtml()


# ============================================
#             Interface Streamlit
# ============================================

st.title("🌩️ Méthode des Nuées Dynamiques (Dynamic Clusters)")
st.write("Implémentation basée sur la méthode originale de Diday (1971)")

# Paramètres utilisateur
K = st.sidebar.slider("Nombre de classes (K)", 2, 8, 3)
ni = st.sidebar.slider("Nombre d'étalons par classe (ni)", 2, 20, 5)
seed = st.sidebar.number_input("Random seed", 0, 9999, 0)

# Charger données
X = load_data()

# Exécuter méthode
model = DynamicClusters(K=K, n_standards=ni, random_state=seed)
model.fit(X)

# Afficher partition finale
fig = plot_results(X, model)
st.pyplot(fig)

# Animation
st.subheader("🎞 Animation de l'évolution des nuées dynamiques")
html_anim = animate_clusters(X, model)
html(html_anim, height=600)
