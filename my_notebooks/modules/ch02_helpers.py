import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class BaseGate:
    """ゲートの抽象基底クラス的な立ち位置（インターフェース用）"""

    def __call__(self, x1, x2):
        raise NotImplementedError

    def get_linear_subgates(self):
        """
        元の入力 (x1, x2) に対して線形境界を持つ Gate のリストを返す
        単純な Gate なら [self]
        CompositeGate は内部の gate1, gate2 から再帰的に集める
        """
        return []


class Gate(BaseGate):
    """
    2入力パーセプトロン型ゲート
    y = 1  (w1*x1 + w2*x2 + b > 0)
      = 0  (それ以外)
    """
    def __init__(self, w, b, name=None):
        self.w = np.array(w, dtype=float)  # [w1, w2]
        self.b = float(b)
        self.name = name or "Gate"

    def __call__(self, x1, x2):
        x = np.array([x1, x2], dtype=float)
        tmp = np.sum(self.w * x) + self.b
        return 0 if tmp <= 0 else 1

    def get_linear_subgates(self):
        # 自分自身が元の入力空間に対する線形境界を持つ
        return [self]


class CompositeGate(BaseGate):
    """
    任意の2つのゲート出力を、さらに任意のゲートで結合する合成ゲート

        y = out_gate( gate1(x1, x2), gate2(x1, x2) )

    gate1, gate2, out_gate はいずれも BaseGate 互換のオブジェクト
    """
    def __init__(self, gate1, gate2, out_gate, name=None):
        self.gate1 = gate1
        self.gate2 = gate2
        self.out_gate = out_gate
        self.name = name or "CompositeGate"

    def __call__(self, x1, x2):
        s1 = self.gate1(x1, x2)
        s2 = self.gate2(x1, x2)
        return self.out_gate(s1, s2)

    def get_linear_subgates(self):
        """
        gate1, gate2 の「元の入力に対する線形のサブゲート」を再帰的に収集
        out_gate は gate1, gate2 の出力空間上の線形境界なので、
        元の (x1, x2) 空間の直線としては描画しない
        """
        sub = []
        for g in (self.gate1, self.gate2):
            if hasattr(g, "get_linear_subgates"):
                sub.extend(g.get_linear_subgates())

        # 重複 Gate を除去（同一インスタンスなら1つにまとめる）
        unique = []
        seen_ids = set()
        for g in sub:
            if id(g) not in seen_ids:
                seen_ids.add(id(g))
                unique.append(g)
        return unique


class DecisionBoundaryPlotter:
    """
    パーセプトロンを用いた論理ゲートの決定境界を可視化するためのクラス

    - コンストラクタで xlim, ylim, figsize, dpi などの共通項目を設定
    - plot(gate) で：
        * gate(x1, x2) を格子状に評価して y=1 の領域を塗りつぶし
        * gate.get_linear_subgates() から取得した線形境界を直線として描画
        * (0,0),(1,0),(0,1),(1,1) の入力に対応する点と、それぞれに対する y の値をマーカーで描画
    """

    def __init__(
        self,
        xlim=(-0.5, 1.5),
        ylim=(-0.5, 1.5),
        figsize=None,
        width_px=None,
        height_px=None,
        dpi=100,
        fontsize=14,
    ):
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize      # (インチ, インチ)
        self.width_px = width_px    # ピクセル指定したい場合
        self.height_px = height_px
        self.dpi = dpi
        self.fontsize = fontsize

    # ---- 内部用のメソッド：Figure を作る ----
    def _create_figure(self):
        if self.figsize is not None:
            # ユーザーが fig サイズ（インチ）を明示指定
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        elif self.width_px is not None and self.height_px is not None:
            # ピクセル指定 + dpi から figsize を計算
            w_in = self.width_px / self.dpi
            h_in = self.height_px / self.dpi
            fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=self.dpi)
        else:
            # デフォルトの fig サイズ + dpi
            fig, ax = plt.subplots(dpi=self.dpi)

        return fig, ax

    # ---- メインの描画メソッド ----
    def plot(self, gate, title=None, cmap=None, show_points=True):
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim

        # 0 の領域を透明に、1 の領域を赤で表示するカラーマップを作成
        if cmap is None:
            colors = ['white', 'red']
            cmap = ListedColormap(colors)

        fig, ax = self._create_figure()

        # --- 軸まわりの体裁 ---
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", "box")

        # x1, x2 の軸 (矢印付き)
        ax.annotate(
            "",
            xy=(x_max, 0),
            xytext=(x_min, 0),
            arrowprops=dict(width=1.0, headwidth=8.0, color="gray"),
            zorder=3,
        )
        ax.annotate(
            "",
            xy=(0, y_max),
            xytext=(0, y_min),
            arrowprops=dict(width=1.0, headwidth=8.0, color="gray"),
            zorder=3,
        )

        ax.text(
            x_max + 0.08,
            0,
            r"$x_1$",
            ha="left",
            va="center",
            fontsize=self.fontsize + 2,
            zorder=4,
        )
        ax.text(
            0,
            y_max + 0.08,
            r"$x_2$",
            ha="center",
            va="bottom",
            fontsize=self.fontsize + 2,
            zorder=4,
        )

        # --- 領域の塗り分け (gate(x1,x2) を各格子点ごとに評価) ---
        nx, ny = 301, 301
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
        )
        gate_vec = np.vectorize(gate)
        zz = gate_vec(xx, yy)  # 0 or 1

        ax.pcolormesh(
            xx,
            yy,
            zz,
            #shading="auto",
            cmap=cmap,
            alpha=0.2,
            zorder=1,
            vmin=0,
            vmax=1,
        )

        # --- サブゲートから決定境界（直線）を描画 ---
        pts = np.linspace(x_min, x_max, 400)
        linear_subgates = []
        if hasattr(gate, "get_linear_subgates"):
            linear_subgates = gate.get_linear_subgates()

        for g_sub in linear_subgates:
            w = g_sub.w
            b = g_sub.b
            # w1*x1 + w2*x2 + b = 0 から x2 = -(w1*x1 + b)/w2
            if abs(w[1]) < 1e-12:
                # x2 の係数が 0 に近いときは縦線 (x1 = const) になる
                # w1*x1 + b = 0 → x1 = -b/w1
                x_const = -b / w[0]
                ax.axvline(x_const, color="red", lw=2, zorder=4)
            else:
                line = -(w[0] * pts + b) / w[1]
                ax.plot(pts, line, color="red", lw=2, zorder=4)

        # --- 入力の座標 (0,0),(1,0),(0,1),(1,1) ---
        if show_points:
            discrete_pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for x1, x2 in discrete_pts:
                y = gate(x1, x2)
                if y == 1:
                    marker = "^"
                    color = "red"
                    facecolor = "none"
                else:
                    marker = "o"
                    color = "black"
                    facecolor = "white"

                ax.scatter(
                    x1,
                    x2,
                    marker=marker,
                    s=80,
                    edgecolor=color,
                    facecolor=facecolor,
                    linewidth=1.5,
                    zorder=5,
                )

                # 座標のラベル
                if (x1, x2) in [(0, 0), (1, 0)]:
                    ax.text(
                        x1,
                        x2 - 0.08,
                        f"({x1}, {x2})",
                        fontsize=self.fontsize,
                        ha="center",
                        va="top",
                        zorder=6,
                    )
                else:
                    ax.text(
                        x1,
                        x2 + 0.08,
                        f"({x1}, {x2})",
                        fontsize=self.fontsize,
                        ha="center",
                        va="bottom",
                        zorder=6,
                    )

        if title is None and hasattr(gate, "name"):
            title = gate.name
        if title is not None:
            ax.set_title(title, fontsize=self.fontsize + 1, pad=12)

        return fig, ax
