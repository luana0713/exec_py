import cv2
import numpy as np


render = np.zeros((540, 960, 3), dtype=np.uint8)
cv2.circle(render, (480, 270), 150, (100, 150, 200), -1)
cv2.rectangle(render, (650, 120), (800, 400), (80, 120, 80), -1)
cv2.imwrite('blender_render0001.png', render)

# --- Ler imagem ---
img = cv2.imread('blender_render0001.png')

if img is None:
    print("Erro ao carregar imagem!")
    exit()

# --- Marca d’água ---
cv2.putText(img, 'lkive |007', (20, 520),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 200), 2)

# --- Bordas ---
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bordas = cv2.Canny(cinza, 30, 100)
bordas_bgr = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

resultado = cv2.addWeighted(img, 0.85, bordas_bgr, 0.15, 0)

# --- Salvar ---
cv2.imwrite('resultado_final.png', resultado)

# --- Mostrar ---
cv2.imshow("Resultado", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Resultado salvo em resultado_final.png")