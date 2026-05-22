import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Analitik DJPb", layout="wide")
st.title("Dashboard Analitik DJPb")

MODEL_PATH = Path("model/Best_model.pkcls")
DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

data = load_data()
model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])

with tab_prediksi:
    # ── Load model (satu kali) ────────────────────────────────────────────────────
    @st.cache_resource
    def load_model():
        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)
    
    model = None
    model_error = None
    try:
        model = load_model()
    except Exception as exc:
        model_error = str(exc)
    
    # ── CSS tambahan ──────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Verdict banner ── */
    .verdict-ya    { background:#eafaf3; border:1px solid #97C459;
                    border-radius:12px; padding:1.1rem 1.3rem; }
    .verdict-tidak { background:#fcebeb; border:1px solid #F09595;
                    border-radius:12px; padding:1.1rem 1.3rem; }
    .verdict-title { font-size:11px; font-weight:600; letter-spacing:.07em;
                    text-transform:uppercase; margin-bottom:2px; }
    .verdict-title-ya    { color:#3B6D11; }
    .verdict-title-tidak { color:#A32D2D; }
    .verdict-val { font-size:26px; font-weight:600; }
    .verdict-val-ya    { color:#27500A; }
    .verdict-val-tidak { color:#791F1F; }
    
    /* ── Section label ── */
    .sec-label { font-size:11px; font-weight:600; letter-spacing:.08em;
                text-transform:uppercase; color:#adb5bd;
                margin-top:1.4rem; margin-bottom:.4rem; }
    
    /* ── Input summary chips ── */
    .chip-grid { display:flex; flex-wrap:wrap; gap:8px; margin-top:.5rem; }
    .chip      { background:#f0f4f8; border-radius:20px;
                padding:4px 12px; font-size:12px; color:#344054; }
    .chip b    { color:#101828; }
    
    /* ── Tipe selector ── */
    div[data-testid="stHorizontalBlock"] .stButton button {
        width:100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # ════════════════════════════════════════════════════════════════════════════
    # LAYOUT UTAMA
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("### Prediksi Realisasi Anggaran 95%")
    st.caption("Atur parameter satker di bawah — hasil prediksi akan muncul otomatis.")
    
    if model_error:
        st.error(f"Model tidak dapat dimuat: {model_error}")
    
    left, right = st.columns([1.1, 1], gap="large")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # KOLOM KIRI — Form input
    # ─────────────────────────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="sec-label">Parameter input</div>', unsafe_allow_html=True)
    
        jumlah_spm = st.slider(
            "📄 Jumlah SPM",
            min_value=1, max_value=200, value=90, step=1,
            help="Surat Perintah Membayar yang diterbitkan sepanjang tahun",
        )
    
        revisi_dipa = st.slider(
            "✏️ Revisi DIPA",
            min_value=0, max_value=5, value=1, step=1,
            help="Jumlah kali revisi Daftar Isian Pelaksanaan Anggaran",
        )
    
        deviasi_rpd = st.slider(
            "📉 Deviasi RPD (%)",
            min_value=0.0, max_value=30.0, value=10.0, step=0.1,
            format="%.1f",
            help="Selisih antara rencana penarikan dana dan realisasi",
        )
    
        skor_ikpa = st.slider(
            "🏅 Skor IKPA",
            min_value=70.0, max_value=100.0, value=85.0, step=0.1,
            format="%.1f",
            help="Indikator Kinerja Pelaksanaan Anggaran",
        )
    
        st.markdown('<div class="sec-label">Tipe satker</div>', unsafe_allow_html=True)
        tipe_satker = st.radio(
            "Tipe satker",
            options=["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
            index=2,
            horizontal=True,
            label_visibility="collapsed",
        )
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # KOLOM KANAN — Hasil prediksi
    # ─────────────────────────────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="sec-label">Hasil prediksi</div>', unsafe_allow_html=True)
    
        tipe_mapping = {
            "Dekonsentrasi":     [1.0, 0.0, 0.0, 0.0],
            "Kantor Daerah":     [0.0, 1.0, 0.0, 0.0],
            "Kantor Pusat":      [0.0, 0.0, 1.0, 0.0],
            "Tugas Pembantuan":  [0.0, 0.0, 0.0, 1.0],
        }
    
        features = [
            float(jumlah_spm),
            float(revisi_dipa),
            float(deviasi_rpd),
            float(skor_ikpa),
            *tipe_mapping[tipe_satker],
        ]
        X = np.array([features])
    
        # ── Jalankan prediksi ──────────────────────────────────────────────────
        if model is None:
            st.info("Model belum dimuat. Tampilan demo menggunakan heuristik sederhana.")
    
            # Heuristik demo agar UI tetap berjalan tanpa model
            score = 0
            if jumlah_spm >= 80:  score += 30
            elif jumlah_spm >= 40: score += 15
            if revisi_dipa <= 2:  score += 15
            if deviasi_rpd <= 10: score += 20
            elif deviasi_rpd <= 20: score += 8
            if skor_ikpa >= 90:   score += 25
            elif skor_ikpa >= 80: score += 12
            if tipe_satker in ("Kantor Pusat", "Tugas Pembantuan"): score += 10
    
            prob_ya    = min(0.95, max(0.05, score / 100))
            prob_tidak = 1 - prob_ya
            label_value = "Ya" if prob_ya >= 0.5 else "Tidak"
    
        else:
            prediction = model.predict(X)
            if isinstance(prediction, tuple):
                prediction_values, probabilities = prediction
            else:
                prediction_values = prediction
                probabilities = None
    
            label_index = int(prediction_values[0])
            label_value = model.domain.class_var.values[label_index]
    
            if probabilities is not None and probabilities.shape[1] > 1:
                prob_ya    = float(probabilities[0, 1])
                prob_tidak = float(probabilities[0, 0])
            else:
                prob_ya    = 1.0 if label_value == "Ya" else 0.0
                prob_tidak = 1 - prob_ya
    
        # ── Verdict banner ──────────────────────────────────────────────────────
        is_ya = label_value == "Ya"
        css_class = "verdict-ya" if is_ya else "verdict-tidak"
        icon_html  = "✅" if is_ya else "❌"
        title_cls  = "verdict-title-ya" if is_ya else "verdict-title-tidak"
        val_cls    = "verdict-val-ya" if is_ya else "verdict-val-tidak"
        verdict_text = "Ya — tercapai" if is_ya else "Tidak — belum tercapai"
    
        st.markdown(f"""
        <div class="{css_class}">
        <div class="verdict-title {title_cls}">Prediksi realisasi 95%</div>
        <div class="verdict-val {val_cls}">{icon_html} {verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("")
    
        # ── Gauge probabilitas ──────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_ya * 100, 1),
            number={"suffix": "%", "font": {"size": 28,
                    "color": "#27500A" if is_ya else "#791F1F"}},
            title={"text": "Probabilitas tercapai",
                "font": {"size": 12, "color": "#6c757d"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                    "tickwidth": 1, "tickcolor": "#dee2e6",
                },
                "bar": {"color": "#639922" if is_ya else "#E24B4A", "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50],  "color": "#fff5f5"},
                    {"range": [50, 100],"color": "#f0faf4"},
                ],
                "threshold": {
                    "line": {"color": "#185FA5", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            height=220,
            margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="white",
            font={"family": "sans-serif"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config={"displayModeBar": False})
    
        # ── Progress bar Ya vs Tidak ────────────────────────────────────────────
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Probabilitas Ya", f"{prob_ya:.1%}")
            st.progress(prob_ya)
        with prob_col2:
            st.metric("Probabilitas Tidak", f"{prob_tidak:.1%}")
            st.progress(prob_tidak)
    
        # ── Ringkasan input ─────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Ringkasan input</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chip-grid">
        <span class="chip">SPM: <b>{jumlah_spm}</b></span>
        <span class="chip">Revisi DIPA: <b>{revisi_dipa}</b></span>
        <span class="chip">Deviasi RPD: <b>{deviasi_rpd:.1f}%</b></span>
        <span class="chip">IKPA: <b>{skor_ikpa:.1f}</b></span>
        <span class="chip">Tipe: <b>{tipe_satker}</b></span>
        </div>
        """, unsafe_allow_html=True)
    
        # ── Ekspander JSON detail ───────────────────────────────────────────────
        with st.expander("Lihat data JSON input"):
            st.json({
                "jumlah_spm": int(jumlah_spm),
                "revisi_dipa": int(revisi_dipa),
                "deviasi_rpd_persen": round(float(deviasi_rpd), 2),
                "skor_ikpa": round(float(skor_ikpa), 2),
                "tipe_satker": tipe_satker,
                "features_vector": [round(f, 2) for f in features],
            })

with tab_visualisasi:
    # ── Page config ───────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Dashboard Realisasi Anggaran",
        page_icon="📊",
        layout="wide",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .section-title {
            font-size: 11px; font-weight: 600; letter-spacing: .08em;
            text-transform: uppercase; color: #adb5bd;
            margin-bottom: .5rem; margin-top: 1.5rem;
        }
        .tipe-header {
            background: #f0f4f8;
            border-left: 4px solid #378ADD;
            border-radius: 6px;
            padding: 8px 14px;
            margin-bottom: 10px;
            font-weight: 600;
            font-size: 14px;
            color: #1a3a5c;
        }
        .speedometer-wrap {
            background: #fff;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 6px 4px 0 4px;
            text-align: center;
        }
        .satker-sub {
            font-size: 11px; color: #6c757d; margin-top: -4px; margin-bottom: 4px;
        }
    </style>
    """, unsafe_allow_html=True)


    # ── Data ─────────────────────────────────────────────────────────────────────
    df_all = load_data()


    # ── Sidebar filters ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🗂️ Filter")

        kem_opts = ["Semua"] + sorted(df_all["nama_kementerian"].unique().tolist())
        selected_kem = st.selectbox("Kementerian", kem_opts)

        tipe_opts = sorted(df_all["tipe_satker"].unique().tolist())
        selected_tipe = st.multiselect("Tipe Satker", tipe_opts, default=tipe_opts)

        belanja_opts = sorted(df_all["jenis_belanja_utama"].unique().tolist())
        selected_belanja = st.multiselect("Jenis Belanja", belanja_opts, default=belanja_opts)

        tw_sel = st.selectbox("Triwulan speedometer", ["TW1","TW2","TW3"], index=2)

        st.markdown("---")
        st.caption("Data contoh — 10 satker")


    # ── Apply filters ─────────────────────────────────────────────────────────────
    df = df_all.copy()
    if selected_kem != "Semua":
        df = df[df["nama_kementerian"] == selected_kem]
    if selected_tipe:
        df = df[df["tipe_satker"].isin(selected_tipe)]
    if selected_belanja:
        df = df[df["jenis_belanja_utama"].isin(selected_belanja)]

    if df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.stop()

    tw_col_map = {"TW1": "realisasi_tw1_persen",
                "TW2": "realisasi_tw2_persen",
                "TW3": "realisasi_tw3_persen"}
    tw_col = tw_col_map[tw_sel]


    # ── Helpers ───────────────────────────────────────────────────────────────────
    def gauge_color(val: float) -> str:
        if val >= 75: return "#1D9E75"
        if val >= 50: return "#378ADD"
        if val >= 30: return "#EF9F27"
        return "#E24B4A"

    def make_speedometer(value: float, title: str, subtitle: str) -> go.Figure:
        color = gauge_color(value)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 18, "color": color}},
            title={"text": f"<b>{title}</b><br><span style='font-size:10px;color:#6c757d'>{subtitle}</span>",
                "font": {"size": 11}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#dee2e6",
                        "tickvals": [0, 25, 50, 75, 100],
                        "ticktext": ["0","25","50","75","100"]},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "#fff5f5"},
                    {"range": [30, 50], "color": "#fff8e8"},
                    {"range": [50, 75], "color": "#e8f4fd"},
                    {"range": [75, 100],"color": "#eafaf3"},
                ],
                "threshold": {
                    "line": {"color": "#dc3545", "width": 2},
                    "thickness": 0.75,
                    "value": 95,
                },
            },
        ))
        fig.update_layout(
            margin=dict(t=60, b=10, l=20, r=20),
            height=200,
            paper_bgcolor="white",
            font={"family": "sans-serif"},
        )
        return fig


    # ══════════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════════
    st.title("📊 Dashboard Realisasi Anggaran")
    st.caption("Monitoring penyerapan anggaran satker per triwulan · Data contoh")

    # ── Ringkasan metrik ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Ringkasan</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Pagu", f"Rp {df['pagu_miliar'].sum():.1f} M", f"{len(df)} satker")
    with c2: st.metric("Rata-rata TW1", f"{df['realisasi_tw1_persen'].mean():.1f}%")
    with c3: st.metric("Rata-rata TW2", f"{df['realisasi_tw2_persen'].mean():.1f}%")
    with c4: st.metric("Rata-rata TW3", f"{df['realisasi_tw3_persen'].mean():.1f}%")
    with c5:
        cap_ya = (df["realisasi_tercapai_95persen"] == "Ya").sum()
        st.metric("Rata-rata IKPA", f"{df['skor_ikpa'].mean():.1f}",
                f"{cap_ya}/{len(df)} tercapai 95%")


    # ══════════════════════════════════════════════════════════════════════════════
    # SPEEDOMETER — GROUPBY TIPE SATKER
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown(
        f'<div class="section-title">Speedometer Realisasi {tw_sel} — per Tipe Satker</div>',
        unsafe_allow_html=True,
    )

    tipe_groups = df.groupby("tipe_satker")

    for tipe_name, grp in tipe_groups:
        n_satker = len(grp)
        avg_tipe = grp[tw_col].mean()

        st.markdown(
            f'<div class="tipe-header">📁 {tipe_name} '
            f'<span style="font-weight:400;font-size:12px;color:#6c757d">'
            f'({n_satker} satker · rata-rata {tw_sel}: {avg_tipe:.1f}%)</span></div>',
            unsafe_allow_html=True,
        )

        # Rata-rata tipe + tiap satker
        items = [("Rata-rata\n" + tipe_name, avg_tipe, f"Avg {n_satker} satker")] + \
                [(row["kode_satker"], row[tw_col], row["provinsi"])
                for _, row in grp.sort_values(tw_col, ascending=False).iterrows()]

        # Render per baris, maks 4 kolom
        COLS = 4
        for batch_start in range(0, len(items), COLS):
            batch = items[batch_start:batch_start + COLS]
            cols = st.columns(len(batch))
            for col, (title, val, sub) in zip(cols, batch):
                with col:
                    fig = make_speedometer(val, title, sub)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # GRAFIK: REALISASI PER JENIS BELANJA
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Realisasi Belanja per Jenis Belanja</div>', unsafe_allow_html=True)

    df_belanja = (
        df.groupby("jenis_belanja_utama")
        .agg(
            pagu=("pagu_miliar", "sum"),
            tw1=("realisasi_tw1_persen", "mean"),
            tw2=("realisasi_tw2_persen", "mean"),
            tw3=("realisasi_tw3_persen", "mean"),
            n=("kode_satker", "count"),
        )
        .reset_index()
        .sort_values("tw3", ascending=False)
    )

    gcol1, gcol2 = st.columns(2)

    # Grouped bar: TW1/TW2/TW3 per jenis belanja
    with gcol1:
        st.subheader("Rata-rata Realisasi (%) per Jenis Belanja")
        df_melt = df_belanja.melt(
            id_vars=["jenis_belanja_utama"],
            value_vars=["tw1","tw2","tw3"],
            var_name="Triwulan", value_name="Realisasi (%)"
        )
        df_melt["Triwulan"] = df_melt["Triwulan"].str.upper()

        fig_grp = px.bar(
            df_melt,
            x="jenis_belanja_utama", y="Realisasi (%)",
            color="Triwulan",
            barmode="group",
            color_discrete_map={"TW1":"#B5D4F4","TW2":"#378ADD","TW3":"#185FA5"},
            text_auto=".1f",
        )
        fig_grp.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(orientation="h", y=1.08, title_text=""),
            yaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#f0f0f0", title=""),
            xaxis_title="",
        )
        fig_grp.update_traces(textposition="outside", textfont_size=9)
        st.plotly_chart(fig_grp, use_container_width=True)

    # Total pagu per jenis belanja
    with gcol2:
        st.subheader("Total Pagu (Miliar) per Jenis Belanja")
        fig_pagu = px.bar(
            df_belanja.sort_values("pagu"),
            x="pagu", y="jenis_belanja_utama",
            orientation="h",
            color="tw3",
            color_continuous_scale=["#F09595","#378ADD","#1D9E75"],
            range_color=[0,100],
            text="pagu",
            labels={"pagu":"Pagu (M)", "jenis_belanja_utama":"", "tw3":"TW3 (%)"},
        )
        fig_pagu.update_traces(texttemplate="Rp %{text:.1f}M", textposition="outside")
        fig_pagu.update_coloraxes(colorbar_title="TW3 %")
        fig_pagu.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=20, l=0, r=0),
            xaxis=dict(gridcolor="#f0f0f0", title="Miliar Rupiah"),
        )
        st.plotly_chart(fig_pagu, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # GRAFIK: SPM & PEGAWAI dalam satu grafik (dual-axis)
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Jumlah SPM & Pegawai per Satker</div>', unsafe_allow_html=True)

    df_spm = df.sort_values("jumlah_spm", ascending=False).reset_index(drop=True)

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    fig_dual.add_trace(
        go.Bar(
            x=df_spm["kode_satker"],
            y=df_spm["jumlah_pegawai"],
            name="Jumlah Pegawai",
            marker_color="#AFA9EC",
            text=df_spm["jumlah_pegawai"],
            textposition="outside",
            textfont_size=10,
        ),
        secondary_y=False,
    )

    fig_dual.add_trace(
        go.Scatter(
            x=df_spm["kode_satker"],
            y=df_spm["jumlah_spm"],
            name="Jumlah SPM",
            mode="lines+markers+text",
            line=dict(color="#D85A30", width=2.5),
            marker=dict(size=8, color="#D85A30", symbol="circle"),
            text=df_spm["jumlah_spm"],
            textposition="top center",
            textfont_size=10,
        ),
        secondary_y=True,
    )

    fig_dual.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=20, l=0, r=0),
        legend=dict(orientation="h", y=1.08, title_text=""),
        xaxis=dict(title="", gridcolor="#f0f0f0"),
        hovermode="x unified",
        height=380,
    )
    fig_dual.update_yaxes(title_text="Jumlah Pegawai", secondary_y=False,
                        gridcolor="#f0f0f0", rangemode="tozero")
    fig_dual.update_yaxes(title_text="Jumlah SPM", secondary_y=True,
                        showgrid=False, rangemode="tozero")

    st.plotly_chart(fig_dual, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # SCATTER: Realisasi TW3 vs IKPA
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Realisasi TW3 vs Skor IKPA</div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(
        df,
        x="realisasi_tw3_persen", y="skor_ikpa",
        color="nama_kementerian",
        size="pagu_miliar",
        hover_data=["kode_satker","provinsi","jenis_belanja_utama","deviasi_rpd_persen"],
        text="kode_satker",
        labels={"realisasi_tw3_persen":"Realisasi TW3 (%)","skor_ikpa":"Skor IKPA",
                "nama_kementerian":"Kementerian","pagu_miliar":"Pagu (M)"},
    )
    fig_scatter.add_hline(y=75, line_dash="dot", line_color="#adb5bd",
                        annotation_text="IKPA minimal (75)")
    fig_scatter.add_vline(x=50, line_dash="dot", line_color="#adb5bd",
                        annotation_text="Target TW3 (50%)")
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20, l=0, r=0),
        xaxis=dict(range=[0,100], gridcolor="#f0f0f0", ticksuffix="%"),
        yaxis=dict(range=[60,102], gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # TABEL DETAIL
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Tabel Detail Satker</div>', unsafe_allow_html=True)

    df_tabel = df[[
        "kode_satker","nama_kementerian","provinsi","tipe_satker",
        "jenis_belanja_utama","pagu_miliar","jumlah_pegawai","jumlah_spm",
        "revisi_dipa","realisasi_tw1_persen","realisasi_tw2_persen",
        "realisasi_tw3_persen","deviasi_rpd_persen","skor_ikpa",
        "realisasi_tercapai_95persen",
    ]].rename(columns={
        "kode_satker":"Kode","nama_kementerian":"Kementerian","provinsi":"Provinsi",
        "tipe_satker":"Tipe","jenis_belanja_utama":"Jenis Belanja",
        "pagu_miliar":"Pagu (M)","jumlah_pegawai":"Pegawai","jumlah_spm":"SPM",
        "revisi_dipa":"Rev. DIPA","realisasi_tw1_persen":"TW1 (%)","realisasi_tw2_persen":"TW2 (%)",
        "realisasi_tw3_persen":"TW3 (%)","deviasi_rpd_persen":"Deviasi RPD (%)",
        "skor_ikpa":"IKPA","realisasi_tercapai_95persen":"95% Tercapai",
    })

    st.dataframe(
        df_tabel.style
            .background_gradient(subset=["TW3 (%)"], cmap="Blues", vmin=0, vmax=100)
            .background_gradient(subset=["IKPA"], cmap="Greens", vmin=60, vmax=100)
            .applymap(
                lambda v: ("color:#155724;background-color:#d4edda;" if v == "Ya"
                        else "color:#721c24;background-color:#f8d7da;" if v == "Tidak" else ""),
                subset=["95% Tercapai"],
            )
            .format({
                "Pagu (M)":"{:.2f}","TW1 (%)":"{:.1f}","TW2 (%)":"{:.1f}",
                "TW3 (%)":"{:.1f}","Deviasi RPD (%)":"{:.2f}","IKPA":"{:.2f}",
            }),
        use_container_width=True,
        height=420,
    )

    st.download_button(
        label="⬇️ Unduh data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="realisasi_anggaran.csv",
        mime="text/csv",
    )
