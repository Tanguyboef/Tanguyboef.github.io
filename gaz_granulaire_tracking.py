# %% [markdown]
# # 1. Paramètres

# %%
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

DEBUG = True

dossier_images = "images"
extensions = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
fps = 30
dt = 1 / fps

largeur_cadre_cm = 40
hauteur_cadre_cm = 40
diametre_bille_cm = 2.4
px_par_cm = 44.65
diametre_bille_px_attendu = 107
nombre_billes_attendu = 50

seuil_detection = None          # None = automatique, sinon ex. 210
cadre_px = None                 # None = auto, sinon (x0, y0, w, h) en pixels
marge_crop_px = 0

distance_max_tracking = 4.0     # cm entre deux images
max_frames_perdues = 5
longueur_min_trajectoire = 8
vitesse_max_cm_s = 150
masse_bille = 1.0               # kg ou unité arbitraire cohérente

tolerance_diametre_px = 0.35
collision_tol_cm = 0.18
fenetre_collision = 2

aire_attendue = np.pi * (diametre_bille_px_attendu / 2) ** 2
aire_min = aire_attendue * 0.45
aire_max = aire_attendue * 1.75

# %% [markdown]
# # 2. Images et recadrage

# %%
def lire_images(dossier):
    p = Path(dossier)
    fichiers = sorted([f for ext in extensions for f in p.glob(ext)])
    if not fichiers:
        raise FileNotFoundError(f"Aucune image JPG dans {p.resolve()}")
    return fichiers

def detecter_cadre(img):
    if cadre_px is not None:
        return tuple(map(int, cadre_px))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (12, 25, 70), (45, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    candidats = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.12 * W * H:
            continue
        ratio = w / h
        centre_ok = 0.25 * W < x + w / 2 < 0.75 * W
        if 0.75 < ratio < 1.35 and centre_ok:
            candidats.append((w * h, x, y, w, h))
    if candidats:
        _, x, y, w, h = max(candidats)
        return x + marge_crop_px, y + marge_crop_px, w - 2 * marge_crop_px, h - 2 * marge_crop_px
    w = int(round(largeur_cadre_cm * px_par_cm)); h = int(round(hauteur_cadre_cm * px_par_cm))
    return max(0, (W - w) // 2), max(0, (H - h) // 2), min(w, W), min(h, H)

def crop_cadre(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w].copy()

# %% [markdown]
# # 3. Détection des billes

# %%
def detecter_billes(crop, i_image):
    gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    thr = int(np.percentile(gris, 96.5)) if seuil_detection is None else seuil_detection
    mask = ((gris >= thr) | ((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > max(170, thr - 15)))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lignes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (aire_min <= area <= aire_max):
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / per**2 if per else 0
        if circ < 0.68:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        d = 2 * r
        if abs(d - diametre_bille_px_attendu) > tolerance_diametre_px * diametre_bille_px_attendu:
            continue
        M = cv2.moments(c)
        if M["m00"]:
            x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
        q = circ * np.exp(-abs(d - diametre_bille_px_attendu) / diametre_bille_px_attendu)
        lignes.append(dict(image=i_image, x_px=x, y_px=y, x_cm=x/px_par_cm, y_cm_haut=y/px_par_cm,
                           y_cm_bas=hauteur_cadre_cm-y/px_par_cm, rayon_px=r, diametre_px=d,
                           aire_px2=area, circularite=circ, qualite=q))
    return pd.DataFrame(lignes).sort_values("qualite", ascending=False).head(nombre_billes_attendu)

def afficher_detection(crop, df, titre=""):
    im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(7, 7)); plt.imshow(im)
    for _, r in df.iterrows():
        plt.scatter(r.x_px, r.y_px, s=18, c="r")
        if "id" in df.columns and pd.notna(r.id):
            plt.text(r.x_px+5, r.y_px+5, int(r.id), color="cyan", fontsize=8)
    plt.title(titre); plt.axis("off"); plt.show()

# %% [markdown]
# # 4. Traitement de toutes les images

# %%
fichiers = lire_images(dossier_images)
img0 = cv2.imread(str(fichiers[0]))
rect_cadre = detecter_cadre(img0)
print("cadre_px détecté =", rect_cadre)

detections = []
for i, f in enumerate(fichiers):
    img = cv2.imread(str(f))
    crop = crop_cadre(img, rect_cadre)
    df = detecter_billes(crop, i)
    df["fichier"] = f.name
    detections.append(df)
    if DEBUG and i == 0:
        print("billes détectées image 0 =", len(df), "diamètre moyen px =", df.diametre_px.mean().round(1))
        afficher_detection(crop, df, "contrôle détection")

detections = pd.concat(detections, ignore_index=True)
detections.to_csv("detections.csv", index=False)
detections.head()

# %% [markdown]
# # 5. Suivi hongrois

# %%
def suivre(dets):
    tracks, next_id, out = {}, 0, []
    for frame, g in dets.groupby("image", sort=True):
        pts = g[["x_cm", "y_cm_haut"]].to_numpy()
        ids_actifs = [tid for tid, t in tracks.items() if frame - t["last"] <= max_frames_perdues]
        assign = {}
        if len(ids_actifs) and len(pts):
            old = np.array([tracks[tid]["pos"] for tid in ids_actifs])
            D = np.linalg.norm(old[:, None, :] - pts[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(D)
            for r, c in zip(rr, cc):
                if D[r, c] <= distance_max_tracking:
                    assign[c] = ids_actifs[r]
        for j, (_, row) in enumerate(g.reset_index(drop=True).iterrows()):
            tid = assign.get(j, next_id)
            if tid == next_id:
                next_id += 1
            tracks[tid] = {"pos": pts[j], "last": frame}
            d = row.to_dict(); d["id"] = tid; out.append(d)
    return pd.DataFrame(out)

trajectoires = suivre(detections)
n = trajectoires.groupby("id").size()
trajectoires = trajectoires[trajectoires.id.isin(n[n >= longueur_min_trajectoire].index)].copy()
trajectoires.to_csv("trajectoires.csv", index=False)
trajectoires.head()

# %% [markdown]
# # 6. Vitesses

# %%
tr = trajectoires.sort_values(["id", "image"]).copy()
tr[["vx_cm_s", "vy_cm_s"]] = tr.groupby("id")[["x_cm", "y_cm_haut"]].diff() / (tr.groupby("id")["image"].diff().to_numpy()[:, None] * dt)
tr["v_cm_s"] = np.hypot(tr.vx_cm_s, tr.vy_cm_s)
vitesses = tr.dropna(subset=["v_cm_s"])
vitesses = vitesses[vitesses.v_cm_s <= vitesse_max_cm_s].copy()
vitesses.to_csv("vitesses.csv", index=False)
vitesses.head()

# %% [markdown]
# # 7. Collisions

# %%
def etat_av_ap(df, tid, frame):
    g = df[df.id == tid].sort_values("image")
    av = g[g.image <= frame - fenetre_collision].tail(1)
    ap = g[g.image >= frame + fenetre_collision].head(1)
    return (None if av.empty else av.iloc[0], None if ap.empty else ap.iloc[0])

cols = []
for frame, g in trajectoires.groupby("image"):
    a = g[["id", "x_cm", "y_cm_haut"]].to_numpy()
    for m in range(len(a)):
        for n in range(m + 1, len(a)):
            d = np.linalg.norm(a[m, 1:3] - a[n, 1:3])
            if abs(d - diametre_bille_cm) <= collision_tol_cm:
                id1, id2 = int(a[m, 0]), int(a[n, 0])
                av1, ap1 = etat_av_ap(vitesses, id1, frame); av2, ap2 = etat_av_ap(vitesses, id2, frame)
                if any(x is None for x in [av1, ap1, av2, ap2]):
                    continue
                v_av = np.array([[av1.vx_cm_s, av1.vy_cm_s], [av2.vx_cm_s, av2.vy_cm_s]])
                v_ap = np.array([[ap1.vx_cm_s, ap1.vy_cm_s], [ap2.vx_cm_s, ap2.vy_cm_s]])
                p_av = masse_bille * v_av.sum(axis=0); p_ap = masse_bille * v_ap.sum(axis=0)
                e_av = 0.5 * masse_bille * np.sum(v_av**2); e_ap = 0.5 * masse_bille * np.sum(v_ap**2)
                cols.append(dict(image=frame, id1=id1, id2=id2, distance_cm=d,
                                 px_av=p_av[0], py_av=p_av[1], px_ap=p_ap[0], py_ap=p_ap[1],
                                 ec_av=e_av, ec_ap=e_ap, perte_energie=e_av-e_ap,
                                 perte_energie_pct=100*(e_av-e_ap)/e_av if e_av else np.nan))
collisions = pd.DataFrame(cols).drop_duplicates(subset=["image", "id1", "id2"])
collisions.to_csv("collisions.csv", index=False)
collisions.head()

# %% [markdown]
# # 8. Graphiques

# %%
if DEBUG:
    crop0 = crop_cadre(cv2.imread(str(fichiers[0])), rect_cadre)
    afficher_detection(crop0, trajectoires[trajectoires.image == 0], "centres et identifiants")

plt.figure(figsize=(7, 7))
for _, g in trajectoires.groupby("id"):
    plt.plot(g.x_cm, hauteur_cadre_cm - g.y_cm_haut, lw=1)
plt.xlim(0, largeur_cadre_cm); plt.ylim(0, hauteur_cadre_cm); plt.gca().set_aspect("equal")
plt.xlabel("x (cm)"); plt.ylabel("y depuis le bas (cm)"); plt.title("trajectoires"); plt.show()

plt.figure(figsize=(6, 4)); plt.hist(vitesses.v_cm_s, bins=30)
plt.xlabel("vitesse (cm/s)"); plt.ylabel("effectif"); plt.title("histogramme des vitesses"); plt.show()

E = vitesses.assign(Ec=0.5 * masse_bille * vitesses.v_cm_s**2).groupby("image").Ec.mean()
plt.figure(figsize=(7, 4)); plt.plot(E.index * dt, E.values)
plt.xlabel("temps (s)"); plt.ylabel("énergie cinétique moyenne"); plt.title("énergie moyenne"); plt.show()
