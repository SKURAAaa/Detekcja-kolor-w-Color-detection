import cv2
import numpy as np
import argparse
import sys
import os


def segmentacja_hsv(klatka):
    # Konwersja do HSV
    hsv = cv2.cvtColor(klatka, cv2.COLOR_BGR2HSV)

    # Czerwony kolor jest na początku i końcu skali Hue
    dolny1 = np.array([0, 140, 70])
    gorny1 = np.array([10, 255, 255])
    dolny2 = np.array([160, 140, 70])
    gorny2 = np.array([180, 255, 255])

    maska1 = cv2.inRange(hsv, dolny1, gorny1)
    maska2 = cv2.inRange(hsv, dolny2, gorny2)
    return cv2.bitwise_or(maska1, maska2)


def usun_szumy(maska):
    kernel = np.ones((5, 5), np.uint8)
    # Opening: usuwa małe kropki (szum)
    maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, kernel)
    # Closing: wypełnia dziury w obiekcie
    maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel)
    return maska


def main():
    # Obsługa video
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Sciezka do pliku wideo")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Blad: Plik {args.video} nie istnieje!")
        sys.exit()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Blad: Nie mozna otworzyc wideo.")
        sys.exit()

    while True:
        ret, frame = cap.read()

        # Pętla video
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        x_center = w // 2

        # Segmentacja i czyszczenie maski
        maska = segmentacja_hsv(frame)
        maska = usun_szumy(maska)

        display_frame = frame.copy()

        # Szukamy konturów,śledzimy tylko pokrywkę (ignorujemy drobny szum)
        contours, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Wybieramy największy obiekt z maski
            najwiekszy_kontur = max(contours, key=cv2.contourArea)

            if cv2.contourArea(najwiekszy_kontur) > 500:  # Sprawdzamy czy obiekt nie jest za mały

                # Momenty i wyznaczanie pozycji DLA KONKRETNEGO OBIEKTU
                M = cv2.moments(najwiekszy_kontur)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Rysowanie okręgu
                    (x, y), radius = cv2.minEnclosingCircle(najwiekszy_kontur)
                    cv2.circle(display_frame, (int(x), int(y)), int(radius), (0, 0, 255), 3)

                    # Odchylenie
                    deviation_px = cx - x_center

                    # Rysowanie osi środka
                    cv2.line(display_frame, (x_center, 0), (x_center, h), (255, 255, 255), 1)

                    # pasek odchylenia
                    bar_y1, bar_y2 = 30, 60
                    if deviation_px > 0:
                        # zielony
                        cv2.rectangle(display_frame, (x_center, bar_y1), (x_center + deviation_px, bar_y2), (0, 255, 0),
                                      -1)
                        kierunek = "Prawa"
                    else:
                        # czerwony
                        cv2.rectangle(display_frame, (x_center + deviation_px, bar_y1), (x_center, bar_y2), (0, 0, 255),
                                      -1)
                        kierunek = "Lewa"

                    cv2.putText(display_frame, f"Dev: {abs(deviation_px)}px ({kierunek})", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # okna
        cv2.imshow("Wideo Oryginalne", display_frame)
        cv2.imshow("Maska Segmentacji", maska)

        # 'q' lub ESC
        if cv2.waitKey(25) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()