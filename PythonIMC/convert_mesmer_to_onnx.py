"""
Convert the pre-trained Mesmer model (.h5/SavedModel) to ONNX format.

Run this ONCE on a machine that has the pythonimc-mesmer conda env (Intel Mac or Linux).
The resulting mesmer.onnx file is saved in models/ and works on any platform via onnxruntime.

Usage (from pythonimc-mesmer env):
    python convert_mesmer_to_onnx.py
    python convert_mesmer_to_onnx.py --output /path/to/mesmer.onnx
"""

import os
import sys
import argparse


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None,
                        help='Output path for mesmer.onnx (default: models/mesmer.onnx)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.output or os.path.join(script_dir, 'models', 'mesmer.onnx')

    try:
        import tensorflow as tf  # type: ignore[import]
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        print("Erreur : TensorFlow introuvable.")
        print("Lance ce script depuis l'env pythonimc-mesmer :")
        print("  conda activate pythonimc-mesmer")
        sys.exit(1)

    try:
        import tf2onnx   # type: ignore[import]
        import onnx      # type: ignore[import]
    except ImportError:
        print("Erreur : tf2onnx ou onnx introuvable.")
        print("  pip install tf2onnx onnx")
        sys.exit(1)

    model_path = os.path.expanduser('~/.keras/models/MultiplexSegmentation')

    if os.path.exists(out_path):
        print(f"mesmer.onnx déjà présent : {out_path}")
        print("Supprime-le et relance pour reconvertir.")
        sys.exit(0)

    if not os.path.exists(model_path):
        print("Téléchargement des poids Mesmer (~97 MB)...")
        from deepcell.applications import Mesmer  # type: ignore[import]
        Mesmer()

    print("Chargement du modèle...")
    model = tf.keras.models.load_model(model_path)
    print(f"  Input : {model.input_shape}")
    print(f"  Outputs: {[o.shape for o in model.outputs]}")

    print("Conversion ONNX (opset 13)...")
    input_sig = [tf.TensorSpec([None, 256, 256, 2], tf.float32, name='input_0')]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_sig, opset=13, output_path=None,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Sauvegarde → {out_path}")
    onnx.save(onnx_model, out_path)

    print(f"Terminé — {os.path.getsize(out_path)/1e6:.1f} MB")
    print("Le fichier fonctionne sur n'importe quelle plateforme avec onnxruntime.")


if __name__ == '__main__':
    main()
