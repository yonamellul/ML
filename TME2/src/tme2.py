import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.base import BaseEstimator

#import pandas as pd


POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


class Density(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
    def fit(self,data):
        pass
    def predict(self,data):
        pass
    def score(self,data):
        density = self.predict(data)
        return np.log(np.where(data==0, 1e-10, data)).sum()

class Histogramme(Density):
	def __init__(self, steps=10):
		Density.__init__(self)
		self.steps = steps
		self.density = None
		self.bins = None

	def fit(self, x):
		"""
		Apprend l'histogramme de la densité sur x
		"""
		self.density, self.bins = np.histogramdd(
			x, bins=[self.steps]*x.shape[-1], density=True
		)

	def predict(self, x):
		def to_bin(x):
			oui = np.stack(self.bins, axis=1)
			l = []
			xi_dim = x.shape[-1]
			for xi in x:
				tmp = []
				for i in range(xi_dim):
					where = np.nonzero(oui[..., i] <= xi[..., i])[-1]
					if where.size == 0:
						tmp.append(None)
					else:
						tmp.append(where[-1] - 1)
				l.append(tuple(tmp))
			return l
		
		prediction = []
		for cords in to_bin(x):
			if cords.count(None) != 0:
				prediction.append(0)
			else:
				prediction.append(self.density[cords])
		return np.array(prediction)

class KernelDensity(Density):
    def __init__(self, kernel=None, sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self, x):
        self.x = x

    def predict(self, data):
        n, d = self.x.shape
        l = []
        for x_i in data:
            x_i_stacked = np.vstack([x_i]*n)
            sum_value = self.kernel((x_i_stacked - self.x)/self.sigma).sum()
            l.append(sum_value / (n*self.sigma**d))
        return np.array(l)

def get_density2D(f, data, steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    xlin, ylin = np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin


def show_density(f, data, steps=100, log=False):
    """
    Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    # extent pour controler l'echelle du plan


def load_poi(typepoi, fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])

    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1], v[1][0][0]]
                    for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data, note



