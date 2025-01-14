import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imshow, imread
from skimage import transform
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin
from skimage import measure
import cv2
import re
from collections import defaultdict
from i_taux_erreur import pytesseract_extract_text

class Character():
    def __init__(self, matrix, description=""):
        self.matrix = matrix
        self.description = description
        self.reduced_vector = None
        
    def resize(self, width:int = 20):
        """Resize en fonction d'une seule dimension seulement"""
        # Calculer la nouvelle largeur pour préserver les proportions
        aspect_ratio = self.matrix.shape[1] / self.matrix.shape[0]
        new_height = int(width * aspect_ratio)
        
        scaled_image = transform.resize(self.matrix, (new_height, width), preserve_range=True)
        
        self.matrix = (scaled_image).astype(np.uint8)
    
    def binarisation(self):
        thresh = threshold_otsu(self.matrix)
        binary = self.matrix > thresh
        self.matrix = 255 * binary
        
    def cadrage(self):
        """https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in
        Plus Llama 3 pour débuggage
        """
        rows = np.any(self.matrix == 0, axis=1)
        cols = np.any(self.matrix == 0, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        self.matrix = self.matrix[ymin:ymax+1, xmin:xmax+1]


    # version 1 modifiée
    def padding(self, dimensions:tuple = (70, 70)):
        """Permet d'avoir une matrice de dimensions fixe en ajoutant des 0
        Codé par Llama 3
        Non testé si l'array self.matrix est plus grand que dimensions ! Peut être ajouter aussi un downscaling dans ce cas
        """
        
        h, w = self.matrix.shape
        """
        if h < dimensions[0] and w > dimensions[1] : 
            resized_image = cv2.resize(self.matrix, (dimensions[0], h), interpolation=cv2.INTER_AREA)
        elif h > dimensions[0] and w < dimensions[1] :
            resized_image = cv2.resize(self.matrix, (w, dimensions[1]), interpolation=cv2.INTER_AREA)
        elif h > dimensions[0] and w > dimensions[1] :
            resized_image = cv2.resize(self.matrix, (dimensions[0], dimensions[1]), interpolation=cv2.INTER_AREA)
        else : 
            resized_image = self.matrix
        
        new_arr = np.zeros(dimensions, dtype=resized_image.dtype)
        """
        new_arr = 255*np.ones(dimensions, dtype=self.matrix.dtype)
        new_arr[:h, :w] = self.matrix
        self.matrix = new_arr
    
    def traitement(self):
        self.binarisation()
        self.cadrage()
        self.padding()
    
    def reduce_dimension(self, pca_instance):
        """A tester. Enregistre le vecteur réduit comme un attribut"""
        self.reduced_vector = pca_instance.transform(np.array([np.ravel(self.matrix)]))[0]
    
    def affiche_matrice(self):
        """Permet d'afficher la matrice"""
        plt.imshow(self.matrix, cmap="gray")
    
    def get_Hu_moments_v2(self):
        """From https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.moments_hu"""
        mu = measure.moments_central(self.matrix)
        nu = measure.moments_normalized(mu)
        return measure.moments_hu(nu)


class Classifieur():
    def __init__(self, n):
        self.pca = PCA(n_components=n)
        self.lettres_references = {}    # Dictionnaire qui associe à un caractère la liste des instances character
        self.reference = {} # Dictionnaire qui associe à un caractère la liste des vecteurs réduits
        self.centers = {}   # Dictionnaire qui associe à un caractère le vecteur moyen des vecteurs réduit de ce caractère

                
    def load_data_degraded(self, folder_path):
        """Rempli le dictionnaire self.lettres_references
        
        Le format des fichiers attendu est "lettre_indice.png" 
        
        """
        self.lettres_references = {}
                
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif')):
                img_path = os.path.join(folder_path, filename)
                img = imread(img_path, as_gray=True)
                # On ajoute dans le dictionnaire
                self.lettres_references[filename.split(".")[0].split("_")[0]] = self.lettres_references.get(filename.split(".")[0], [])
                self.lettres_references[filename.split(".")[0].split("_")[0]].append(Character(img, filename.split(".")[0]))

        # Pour toutes les lettres, on effectue le traitement
        for lst_lettres in self.lettres_references.values():
            for lettre in lst_lettres:
                lettre.traitement()
    
    def generate_data_set(self):
        """Formatte les données pour le PCA. Renvoie un array de la forme (n_samples, n_features)"""
        data_set = []
        for lst_lettres in self.lettres_references.values():
            for lettre in lst_lettres:
                data_set.append(np.ravel(lettre.matrix))
        return np.array(data_set)

    def train(self):
        """Génère le data set et entraîne le PCA"""
        data_set = self.generate_data_set()
        self.pca.fit(data_set)

        for lettre, lst_lettres in self.lettres_references.items():
            for instance_character in lst_lettres:
                # On réduit de dimension la matrice du caractère. Le vecteur réduit est stocké dans l'instance
                # de la classe Character
                instance_character.reduce_dimension(self.pca)
                
                # On enregistre dans le dico self.reference
                self.reference[lettre] = self.reference.get(lettre, [])
                self.reference[lettre].append(instance_character.reduced_vector)
        
    
    def generate_center_dict(self):
        for lettre, lst_vecteur in self.reference.items():
            arrays = []
            for vecteur in lst_vecteur:
                arrays.append(vecteur)
            vecteur_moyen = np.mean(np.array(arrays), axis=0)
            self.centers[lettre] = vecteur_moyen
    
    def compare(self, unknown):
        """Dis de quel centre le caractère est le plus proche"""
        # On applique les traitements au caractère inconnu
        unknown.traitement()

        # On calcule le vecteur réduit le caractérisant
        unknown.reduce_dimension(self.pca)
        
        # On trouve de quel centre il est le plus proche
        Y = np.array([vector for vector in self.centers.values()])
        
        most_near_center_vector = Y[pairwise_distances_argmin(np.array([unknown.reduced_vector]), Y)[0]]
        
        for x, y in self.centers.items():
            if np.all(y == most_near_center_vector):
                return x


def tri_images_dossier_caracteres():
    """
    parcourt toutes les images du dossier TEST/caracteres et ne garde que celles dont le nom 
    est de la forme region{region_count}_ligne{ligne_count}_{caract_count}.bmp, 
    puis les trie par région et par ligne

    Input : 
        - None
    Outpul : 
        - dico_images (dict) : dictionnaire des images trié par régions, lignes et caractères
    """

    # Chemin du dossier à parcourir
    dossier = 'TEST/caracteres'

    # Expression régulière pour matcher les noms de fichiers de la forme 'region{region_count}_ligne{ligne_count}_{caract_count}.bmp'
    pattern = re.compile(r'region(\d+)_ligne(\d+)_(\d+)\.bmp')

    # Dictionnaire pour stocker les images triées par région et ligne
    images_dict = defaultdict(lambda: defaultdict(list))

    # Parcourir le dossier
    for fichier in os.listdir(dossier):
        match = pattern.match(fichier)
        if match:
            region_count = int(match.group(1))
            ligne_count = int(match.group(2))
            caract_count = int(match.group(3))
            
            # Ajouter le chemin complet de l'image à la structure de données
            chemin_image = os.path.join(dossier, fichier)
            images_dict[region_count][ligne_count].append((caract_count, chemin_image))

    # Trier les images par region_count, ligne_count et caract_count
    for region in images_dict:
        for ligne in images_dict[region]:
            images_dict[region][ligne].sort()

    # # Afficher les images triées par région et ligne
    # for region in sorted(images_dict):
    #     print(f"Région {region}:")
    #     for ligne in sorted(images_dict[region]):
    #         print(f"  Ligne {ligne}:")
    #         for caract_count, img in images_dict[region][ligne]:
    #             print(f"    Caractère {caract_count}: {img}")

    return images_dict



def reconnaissance_text_image(classifieur) : 
    """
    Exécute l'algo de reconnaissance de caractères sur toute l'image

    Input : 
        - input_image_path : le chemin de l'image d'entrée.
    Output : 
        - texte (dict) : le texte reconnu, sous format dictionnaire
        - taux (flaot) : le taux d'erreur de notre algo de reconnaissance par rapport à Pytesseract
    """

    # on utilise defaultdict pour créer un dictionnaire de dictionnaires
    # utile lorsqu'on a besoin de créer automatiquement des sous-dictionnaires imbriqués 
    # sans vérifier manuellement leur existence.
    texte_dico = defaultdict(lambda: defaultdict(dict))
    nombre_caract = 0
    caract_identiques = 0

    images_dict = tri_images_dossier_caracteres()

    # parcours des régions
    for region in sorted(images_dict): 
        print()
        print('-------------------------------------------------------------')
        print(f"Région {region}:")

        # parcours des lignes
        for ligne in sorted(images_dict[region]):
            print()
            print(f"  Ligne {ligne}:")

            texte_ligne = ''
            # parcours des caractères
            for caract_count, chemin_image in images_dict[region][ligne]:
                # Ouverture d'une lettre
                im = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
                a = Character(im, "")
                a.traitement()

                # identification
                lettre_identifiee = classifieur.compare(a)
                print(lettre_identifiee, end='')
                texte_ligne += lettre_identifiee
                
                # calcul taux erreur 
                caract_pytesseract = pytesseract_extract_text(chemin_image)
                nombre_caract+=1
                if caract_pytesseract == a : 
                    caract_identiques +=1

            texte_dico[region][ligne]=texte_ligne

    taux = caract_identiques/nombre_caract
    return texte_dico, taux

if __name__=="main" : 
    c = Classifieur(20)
    c.load_data_degraded("LETTRES/ARIAL/Alphabet_arial_minuscule")
    c.train()
    c.generate_center_dict()

    # Ouverture d'une lettre
    im = imread("t.png", as_gray=True)
    a = Character(im, "")
    a.traitement()
    print(c.compare(a))

