import cv2
import numpy as np
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Configurações de segurança do PyAutoGUI (jogue o mouse para o canto da tela para travar)
pyautogui.FAILSAFE = True

st.set_page_config(
    page_title="Controle de Mouse por Visão Computacional", page_icon="🖱️", layout="centered"
)

st.title("🖱️ Controle do Mouse com Movimento das Mãos (YOLOv8)")
st.write(
    "Utilize a sua mão detectada pela webcam para controlar o cursor do mouse na tela."
)

# Barra lateral com controles
st.sidebar.header("Configurações")
confianca = st.sidebar.slider("Confiança Mínima", 0.1, 1.0, 0.5, 0.05)
suavizacao = st.sidebar.slider("Fator de Suavização do Mouse", 1, 10, 3)

# Botões de controle de estado no Streamlit
if "executando" not in st.session_state:
    st.session_state.executando = False


col1, col2 = st.sidebar.columns(2)
if col1:
    if st.button("Iniciar Câmera"):
        st.session_state.executando = True
if col2:
    if st.button("Parar Câmera"):
        st.session_state.executando = False

# Espaço para exibir o vídeo na interface do Streamlit
quadro_video = st.empty()

if st.session_state.executando:
    # Carrega o modelo YOLOv8 Pose (baixado automaticamente na primeira execução)
    @st.cache_resource
    def carregar_modelo():
        return YOLO("yolov8n-pose.pt")


    model = carregar_modelo()

    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()

    # Variáveis para suavização de movimento
    prev_x, prev_y = 0, 0

    st.sidebar.success(
        "Câmera ativa! Mova a mão para controlar o mouse. (Pressione 'Parar Câmera' para encerrar)."
    )

    while st.session_state.executando:
        ret, frame = cap.read()
        if not ret:
            st.error("Não foi possível acessar a webcam.")
            break

        # Espelha a imagem para uma experiência mais natural (tipo espelho)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Executa a inferência YOLOv8 Pose na CPU
        resultados = model(frame, conf=confianca, device="cpu", verbose=False)

        # Processa os keypoints (pontos articulares)
        for r in resultados:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                # Pegamos o primeiro indivíduo detectado
                keypoints = r.keypoints.xy[0].cpu().numpy()

                # No YOLOv8-pose, os índices 9 e 10 correspondem aos pulsos (direito e esquerdo)
                # Vamos usar o pulso direito (índice 9) se disponível, senão o esquerdo (índice 10)
                pulso_x, pulso_y = 0, 0
                if len(keypoints) > 10 and (
                    keypoints[9][0] > 0 or keypoints[9][1] > 0
                ):
                    pulso_x, pulso_y = keypoints[9]
                elif len(keypoints) > 10 and (
                    keypoints[10][0] > 0 or keypoints[10][1] > 0
                ):
                    pulso_x, pulso_y = keypoints[10]

                if pulso_x > 0 and pulso_y > 0:
                    # Mapeia as coordenadas da câmera para a resolução da tela do computador
                    target_x = np.interp(pulso_x, [100, w - 100], [0, screen_w])
                    target_y = np.interp(pulso_y, [100, h - 100], [0, screen_h])

                    # Aplica suavização linear para evitar movimentos bruscos/trêmulos
                    curr_x = prev_x + (target_x - prev_x) / suavizacao
                    curr_y = prev_y + (target_y - prev_y) / suavizacao

                    # Move o mouse via PyAutoGUI
                    try:
                        pyautogui.moveTo(curr_x, curr_y, duration=0.0)
                    except Exception:
                        pass

                    prev_x, prev_y = curr_x, curr_y

                    # Desenha um círculo visual no pulso rastreado
                    cv2.circle(
                        frame,
                        (int(pulso_x), int(pulso_y)),
                        10,
                        (0, 255, 0),
                        -1,
                    )

        # Converte a imagem de BGR para RGB para exibição correta no Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        quadro_video.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True,
            caption="Rastreamento Ativo",
        )

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info(
        "Clique no botão **'Iniciar Câmera'** na barra lateral para começar a interagir."
    )