import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import imutils
from keras.models import load_model


################################################################################################################################################################



# Fonction de transformation morphologique
def morph_transform(ref, test): 
    img1 = test
    img2 = ref 
    height, width = img2.shape[:2]
    orb_detector = cv2.ORB_create(5000) 
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = matcher.match(d1, d2) 
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
    transformed_img = cv2.warpPerspective(test, homography, (width, height)) 
    return transformed_img


################################################################################################################################################################

def process_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    
    ref_test = morph_transform(image1, image2)
    image2 = ref_test
    image1 = cv2.medianBlur(image1, 5)
    image2 = cv2.medianBlur(image2, 5)
    image_res = cv2.bitwise_xor(image1, image2)
    image_res = cv2.medianBlur(image_res, 5)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    image_res = cv2.morphologyEx(image_res, cv2.MORPH_CLOSE, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_res = cv2.morphologyEx(image_res, cv2.MORPH_OPEN, kernel2)
    _, image_res = cv2.threshold(image_res, 125, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(image_res, 30, 200) 
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Traitement des contours
    CX = []
    CY = []
    C = []

    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            CX.append(cx)
            CY.append(cy)
            C.append((cx, cy))

    # Chargement du modèle Keras
    model = load_model('SavePath')

    # Définition des classes
    classes = {
        0: "Open",
        1: "Short",
        2: "Mousebite",
        3: "Spur",
        4: "Copper",
        5: "Pin-Hole"
    }

    pred = []
    confidence = []
    for c in C:
        im1 = Image.open(image2_path).convert('L').crop((c[0]-32, c[1]-32, c[0]+32, c[1]+32))
        im1 = np.array(im1)
        if len(im1.shape) == 2:
            im1 = np.expand_dims(im1, axis=2)
        im1 = np.expand_dims(im1, axis=0)
        a = model.predict(im1, verbose=1, batch_size=1)
        pred.append(np.argmax(a))
        confidence.append(a)

    # Affichage des résultats dans Tkinter
    img2 = cv2.imread(image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_pil = Image.fromarray(img2)
    draw = ImageDraw.Draw(img2_pil)
    font = ImageFont.load_default()
    for i, txt in enumerate(pred):
        draw.text((CX[i], CY[i]), f"{classes[txt]} {confidence[i][0][txt]}", fill='white', font=font)
        draw.rectangle((CX[i]-2, CY[i]-2, CX[i]+2, CY[i]+2), fill='white')

    # Redéfinir la taille de police avec une taille plus grande
    font_large = ImageFont.truetype("arial.ttf", 20)

    for i, txt in enumerate(pred):
        draw.text((CX[i], CY[i]), f"{classes[txt]} {confidence[i][0][txt]}", fill='white', font=font)
        draw.rectangle((CX[i]-2, CY[i]-2, CX[i]+2, CY[i]+2), fill='white')

    # Redéfinir la taille de police avec une taille plus grande
    font_large = ImageFont.truetype("arial.ttf", 90)

    for i, txt in enumerate(pred):
        draw.text((CX[i], CY[i]), f"{classes[txt]}", fill='white', font=font_large)




    # Redimensionnement de l'image pour la minimiser
    img2_pil.thumbnail((550, 2400), Image.LANCZOS)

    # Convertir l'image PIL en objet Tkinter
    img2_tk = ImageTk.PhotoImage(img2_pil)
    result_label.config(image=img2_tk)
    result_label.image = img2_tk



################################################################################################################################################################



# Créer une interface utilisateur Tkinter

root = tk.Tk()
root.title("Image Processing Interface")
root.config(bg="#F8F8F8")

# Titre de l'application
label = tk.Label(root, text="Printed Circuit Board (PCB) defects detection", font=("Arial", 18, "bold"), fg="#071330", bg="white", relief=tk.RAISED, padx=20, pady=10, highlightbackground="#C3CEDA", highlightthickness=2)
label.grid(row=0, column=0, columnspan=3, sticky="ew")

# Image
image_path = "C:\\Users\\hp\\Desktop\\projet indus\\PCB_DATASET\\DATASET\\images\\Open_circuit\\01_open_circuit_02.jpg"
image = Image.open(image_path)
image = image.resize((200, 200), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.image = photo
image_label.grid(row=1, column=0, padx=10, pady=10)

# Texte
paragraph_text = "Une carte de circuit imprimé, ou PCB (Printed Circuit Board), consiste en une plaque de base qui accueille physiquement et par câblage les composants montés en surface (CMS) et raccordés que l'on retrouve dans la majorité des dispositifs électroniques, dans les applications qui requièrent des tracés conducteurs fins, notamment sur les ordinateurs."
paragraph_label = tk.Label(root, text=paragraph_text, font=("Arial", 14), wraplength=400, justify=tk.LEFT, fg="black", bg="#F8F8F8")
paragraph_label.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

# Champ de saisie pour l'image de référence
image1_label = tk.Label(root, text="Reference image:", font=("Arial", 16), bg="#F8F8F8")
image1_label.grid(row=2, column=0, padx=10, pady=10)
image1_entry = tk.Entry(root, width=70)
image1_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
image1_button = tk.Button(root, text="Upload Image", bg="#FF4500", fg="white", command=lambda: image1_entry.insert(tk.END, filedialog.askopenfilename()), font=("Arial", 14))
image1_button.grid(row=2, column=2, padx=10, pady=10)

# Champ de saisie pour l'image de test
image2_label = tk.Label(root, text="Testing image:", font=("Arial", 16), bg="#F8F8F8")
image2_label.grid(row=3, column=0, padx=10, pady=10)
image2_entry = tk.Entry(root, width=70)
image2_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
image2_button = tk.Button(root, text="Upload Image", bg="#FF4500", fg="white", command=lambda: image2_entry.insert(tk.END, filedialog.askopenfilename()), font=("Arial", 14))
image2_button.grid(row=3, column=2, padx=10, pady=10)

# Bouton pour traiter les images
process_button = tk.Button(root, text="Process Images", bg="blue", fg="white", command=lambda: process_images(image1_entry.get(), image2_entry.get()), font=("Arial", 14))
process_button.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

# Étiquette pour afficher les résultats
result_label = tk.Label(root)
result_label.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

# Configurer les poids des colonnes pour qu'elles s'étirent
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)

# Configurer les poids des lignes pour qu'elles s'étirent
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)

# Lancer la boucle principale Tkinter
root.mainloop()

