#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Virtual Painter 4.1
---------------------
•  Mano derecha  
   –  Índice + medio: seleccionar color / CLEAR  
   –  Solo índice: pintar  
•  Mano izquierda  
   –  Pinza pulgar + índice (medio abajo): ajustar grosor con gran precisión  
   –  **Solo índice extendido:** controla grosor deslizando verticalmente (arriba = fino, abajo = grueso)  
•  Colores 100 % opacos.  
•  Filtro avanzado: Gradiente morfológico sobre el vídeo.

Autor: <tu nombre>
"""

import cv2
import mediapipe as mp
import numpy as np
import sys

# ---------------- Parámetros ----------------
THICKNESS_MIN, THICKNESS_MAX = 2, 40        # rango permitido
SMOOTHING_FACTOR = 0.7                      # suavizado exponencial (0-1)
HEADER_H, HEADER_W = 100, 640
COLORS = {
    "CLEAR":  (255, 255, 255),
    "BLUE":   (255,   0,   0),
    "GREEN":  (  0, 255,   0),
    "RED":    (  0,   0, 255),
    "YELLOW": (  0, 255, 255)
}

# -------------- MediaPipe util --------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hands(frame, pipeline):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pipeline.process(rgb)
    if not results.multi_hand_landmarks:
        return []
    hands = []
    for lm, handed in zip(results.multi_hand_landmarks,
                          results.multi_handedness):
        label = handed.classification[0].label  # 'Left' / 'Right'
        hands.append((lm, label))
    return hands


def fingers_up(landmarks, frame_h):
    ids_tips = [4, 8, 12, 16, 20]
    ids_pip  = [3, 6, 10, 14, 18]
    state = []
    for tip, pip in zip(ids_tips, ids_pip):
        state.append(landmarks.landmark[tip].y * frame_h <
                     landmarks.landmark[pip].y * frame_h)
    return state  # [pulgar, índice, medio, anular, meñique]

# ---------------- Interfaz ------------------

def draw_header(img):
    header = np.zeros((HEADER_H, HEADER_W, 3), dtype=np.uint8) + 50
    section_w = HEADER_W // len(COLORS)
    rois = {}
    for i, (name, bgr) in enumerate(COLORS.items()):
        x1, x2 = i * section_w, (i + 1) * section_w
        cv2.rectangle(header, (x1, 0), (x2, HEADER_H),
                      bgr, cv2.FILLED if name != "CLEAR" else -1)
        cv2.rectangle(header, (x1, 0), (x2, HEADER_H), (0, 0, 0), 2)
        cv2.putText(header, name, (x1 + 10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
                    cv2.LINE_AA)
        rois[name] = (x1, x2)
    img[0:HEADER_H, 0:HEADER_W] = header
    return rois


def select_color(x, rois):
    for name, (x1, x2) in rois.items():
        if x1 < x < x2:
            return name
    return None


# -------------- Filtro avanzado -------------

def gradient_filter(gray):
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)


# ------------------- Main -------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara", file=sys.stderr)
        return

    canvas = None
    current_color = "BLUE"
    brush_thickness = 7               # valor inicial
    prev_point_right = None

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if canvas is None:
                canvas = np.zeros_like(frame)

            # --- detección de manos ---
            for lm, label in detect_hands(frame, hands):
                up = fingers_up(lm, h)
                x_tip = int(lm.landmark[8].x * w)
                y_tip = int(lm.landmark[8].y * h)

                # ===== Mano izquierda =====
                if label == 'Left':
                    # 1) Pinza (pulgar + índice, medio abajo) → ajuste fino
                    if up[0] and up[1] and not up[2]:
                        thumb_x = int(lm.landmark[4].x * w)
                        thumb_y = int(lm.landmark[4].y * h)
                        dist = np.hypot(thumb_x - x_tip, thumb_y - y_tip)
                        target = np.interp(dist, [15, 300],
                                           [THICKNESS_MIN, THICKNESS_MAX])
                        brush_thickness = int(SMOOTHING_FACTOR * brush_thickness +
                                              (1 - SMOOTHING_FACTOR) * target)
                        cv2.circle(frame, (x_tip, y_tip), brush_thickness,
                                   COLORS[current_color], 2)

                    # 2) Solo índice arriba → slider vertical
                    elif up[1] and not up[0] and not up[2]:
                        target = np.interp(y_tip, [0, h],
                                           [THICKNESS_MIN, THICKNESS_MAX])
                        brush_thickness = int(SMOOTHING_FACTOR * brush_thickness +
                                              (1 - SMOOTHING_FACTOR) * target)
                        cv2.circle(frame, (x_tip, y_tip), brush_thickness,
                                   COLORS[current_color], 2)

                # ===== Mano derecha =====
                elif label == 'Right':
                    # Selección de color
                    if up[1] and up[2]:
                        prev_point_right = None
                        if y_tip < HEADER_H:
                            name = select_color(x_tip, rois)
                            if name == "CLEAR":
                                canvas[:] = 0
                            elif name:
                                current_color = name
                        cv2.circle(frame, (x_tip, y_tip), 15,
                                   COLORS[current_color], -1)

                    # Pintar
                    elif up[1] and not up[2]:
                        if prev_point_right is None:
                            prev_point_right = (x_tip, y_tip)
                        cv2.line(canvas, prev_point_right, (x_tip, y_tip),
                                 COLORS[current_color], brush_thickness,
                                 cv2.LINE_AA)
                        prev_point_right = (x_tip, y_tip)
                    else:
                        prev_point_right = None

                # Dibujar esqueleto para depurar
                mp_drawing.draw_landmarks(frame, lm,
                                          mp_hands.HAND_CONNECTIONS)

            # --- filtro de contornos ---
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = gradient_filter(gray)
            frame = cv2.addWeighted(frame, 0.8, edges, 0.2, 0)

            # --- combinación opaca lienzo + vídeo ---
            mask_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask_inv = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY_INV)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            frame = cv2.bitwise_or(frame_bg, canvas)

            # --- barra de herramientas ---
            rois = draw_header(frame)

            cv2.imshow("AI Virtual Painter", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()