import os
import argparse
import onnx
from onnx import helper

parser = argparse.ArgumentParser(description="Verify quantized model")
parser.add_argument(
    "--model", type=str, default="mnist_cnn.onnx", help="Path to ONNX model file"
)
args = parser.parse_args()

# モデルをロード
model = onnx.load(args.model)

# グラフ内のノードを走査
for node in model.graph.node:
    if node.domain == "ai.onnx.contrib":
        # 属性をチェック
        print(f"Found QuantizeLinear node: {node.name}, Attributes: {node.attribute}")

        # 新しい属性リストを作成（bit_width と pos を除外）
        new_attributes = []
        for attr in node.attribute:
            if attr.name not in ["bit_width", "pos"]:
                new_attributes.append(attr)

        # 属性リストをクリアし、新しい属性を追加
        del node.attribute[:]  # リストをクリア
        node.attribute.extend(new_attributes)

        # ドメインを ai.onnx に変更
        node.domain = ""  # 標準の ai.onnx ドメインに変更

# モデルの opset を確認・更新
found_contrib = False
for opset in model.opset_import:
    if opset.domain == "ai.onnx.contrib":
        found_contrib = True
        opset.version = 1  # 適切なバージョンに設定
if not found_contrib:
    model.opset_import.append(helper.make_opsetid("ai.onnx.contrib", 1))

# モデルを検証
onnx.checker.check_model(model)
print("Model is valid after modification!")

# 修正したモデルを保存
base, ext = os.path.splitext(args.model)
output_path = f"{base}_mod{ext}"
onnx.save(model, output_path)
