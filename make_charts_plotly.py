"""Plotly charts (static PNG for Medium + interactive HTML) from measured A100 numbers.
Large model: d_model=4096, 32 heads, fp32, one attention layer. Run: python make_charts_plotly.py"""
import os
import plotly.graph_objects as go

OUT = os.path.dirname(os.path.abspath(__file__))
M = ["MHA", "GQA", "MQA", "MLA"]
COL = {"MHA": "#F28E2B", "GQA": "#4E79A7", "MQA": "#59A14F", "MLA": "#B07AA1"}
FONT = dict(family="Inter, Arial, sans-serif", size=16, color="#222")


def style(fig, title, sub, h=520, w=900):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:14px;color:#777'>{sub}</span>", x=0.02, y=0.93),
        template="simple_white", font=FONT, width=w, height=h, margin=dict(t=110, l=70, r=30, b=90),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.22, x=0),
    )
    fig.update_yaxes(gridcolor="#eee", showline=False, ticks="")
    fig.update_xaxes(showline=True, linecolor="#999")
    return fig


def save(fig, name):
    fig.write_image(os.path.join(OUT, name + ".png"), scale=2)
    fig.write_html(os.path.join(OUT, name + ".html"), include_plotlyjs="cdn")
    print("wrote", name)


# 1. KV cache size -------------------------------------------------------
kv = [264160 / 1024, 66040 / 1024, 8255 / 1024, 16510 / 1024]
fig = go.Figure(go.Bar(x=M, y=kv, marker_color=[COL[m] for m in M],
                       text=[f"{v:.1f} MiB<br>({kv[0]/v:.0f}x smaller)" if v != kv[0] else f"{v:.0f} MiB<br>(baseline)" for v in kv],
                       textposition="outside", cliponaxis=False))
fig.update_yaxes(title="KV cache per sequence (MiB)", range=[0, 330])
save(style(fig, "How much memory does each method remember with?",
           "KV cache for one attention layer, 8,192-token prompt, batch 1 (exact sizes)"), "chart_kv_size")

# 2. Decode time per token vs context -------------------------------------
itl2k = [0.649, 0.682, 0.642, 1.013]
itl8k = [0.959, 1.068, 0.680, 1.052]
fig = go.Figure()
for m, a, b in zip(M, itl2k, itl8k):
    fig.add_trace(go.Scatter(x=["2,048 tokens", "8,192 tokens"], y=[a, b], mode="lines+markers+text",
                             name=m, line=dict(color=COL[m], width=4), marker=dict(size=12),
                             text=["", f"{m}  {b:.2f}"], textposition={"GQA": "top right", "MLA": "bottom right"}.get(m, "middle right"), cliponaxis=False))
fig.update_yaxes(title="ms per generated token (lower is better)", range=[0.5, 1.2])
fig.update_xaxes(title="Prompt length", range=[-0.2, 1.5])
fig.update_layout(showlegend=False)
save(style(fig, "MHA and GQA slow down as context grows; MQA doesn't",
           "Decode time per token, A100, batch 1, d_model 4096. MHA +48%, GQA +57%, MQA +6%, MLA +4%"), "chart_decode_itl")

# 3. Prefill TTFT ----------------------------------------------------------
tt = [173.6, 151.8, 145.5, 322.5]
fig = go.Figure(go.Bar(x=M, y=tt, marker_color=[COL[m] for m in M], text=[f"{v:.0f} ms" for v in tt],
                       textposition="outside", cliponaxis=False))
fig.update_yaxes(title="Time to first token (ms, lower is better)", range=[0, 380])
save(style(fig, "Prefill barely cares about the cache; MLA's extra rebuild shows",
           "8,192-token prompt, A100, batch 1"), "chart_prefill_ttft")

# 4. Prefill vs decode, cost per token ------------------------------------
cats = ["Small model<br>128-token prompt", "Large model<br>2,048-token prompt", "Large model<br>8,192-token prompt"]
pre = [0.613 / 128 * 1000, 27.216 / 2048 * 1000, 173.584 / 8192 * 1000]
dec = [646, 649, 959]
fig = go.Figure()
fig.add_trace(go.Bar(x=cats, y=pre, name="Prefill (per prompt token)", marker_color="#4285F4",
                     text=[f"{v:.1f} µs" for v in pre], textposition="outside"))
fig.add_trace(go.Bar(x=cats, y=dec, name="Decode (per generated token)", marker_color="#DB4437",
                     text=[f"{v:.0f} µs<br><b>{v/p:.0f}x</b>" for v, p in zip(dec, pre)], textposition="outside"))
fig.update_yaxes(title="microseconds per token (log scale)", type="log", range=[0, 3.5], tickvals=[1, 10, 100, 1000], ticktext=["1", "10", "100", "1000"])
fig.update_layout(barmode="group")
save(style(fig, "Writing a token costs 45 to 135x more than reading one",
           "MHA on A100, batch 1. Same weights, same layer. Prefill = TTFT / prompt length; decode = ITL"), "chart_prefill_vs_decode")
