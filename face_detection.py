"""Détection de visages en temps réel avec OpenCV (Haar Cascade).

Touches :
    q  -> quitter
    s  -> enregistrer une capture d'écran
"""

import argparse
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Détection de visages en temps réel avec OpenCV"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="indice de la caméra (0 par défaut)")
    parser.add_argument("--scale", type=float, default=1.1,
                        help="scaleFactor de detectMultiScale (1.1 par défaut)")
    parser.add_argument("--neighbors", type=int, default=5,
                        help="minNeighbors : plus élevé = moins de faux positifs")
    parser.add_argument("--min-size", type=int, default=60,
                        help="taille minimale d'un visage en pixels")
    return parser.parse_args()


def main():
    args = parse_args()

    # Classificateur de visages fourni avec OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise SystemExit("Impossible de charger le classificateur Haar Cascade.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la caméra {args.camera}.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Plus d'image reçue de la caméra.")
            break

        # Prétraitement : niveaux de gris + égalisation de l'histogramme
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Détection des visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=(args.min_size, args.min_size),
        )

        # Dessin des rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Visage", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calcul des FPS (lissé)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.putText(frame, f"Visages : {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS : {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detection de visages - OpenCV", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(name, frame)
            print(f"Capture enregistrée : {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
