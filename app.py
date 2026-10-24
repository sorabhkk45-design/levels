#!/usr/bin/env python3
"""Levels: one-tap data, indicators, news, entry timing, back-test and AI view. Pure Python, no pip installs."""
import hmac, json, os, time, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MODELS = list(dict.fromkeys([MODEL, "gemini-flash-latest", "gemini-2.0-flash"]))
PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
PIN = os.environ.get("APP_PIN", "")

# key: (name, Yahoo symbol, news query, option strike step; 0 = no strike hint)
INSTR = {
    "NIFTY": ("Nifty 50", "^NSEI", "Nifty 50 stock market India", 50),
    "SENSEX": ("Sensex", "^BSESN", "Sensex stock market India", 100),
    "BANKNIFTY": ("Bank Nifty", "^NSEBANK", "Bank Nifty India", 100),
    "GOLD": ("Gold", "GC=F", "gold price MCX India", 0),
    "SILVER": ("Silver", "SI=F", "silver price MCX India", 0),
    "CRUDE": ("Crude Oil", "CL=F", "crude oil price MCX India", 0),
}


def http_get(url, headers=None, data=None, timeout=20):
    h = {"User-Agent": "Mozilla/5.0"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_prices(sym):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%s?range=5y&interval=1d" % urllib.parse.quote(sym)
    r = json.loads(http_get(url, timeout=25))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    rows = [(h, l, c, o, t) for h, l, c, o, t in zip(q["high"], q["low"], q["close"], q["open"], r["timestamp"])
            if None not in (h, l, c, o)]
    if len(rows) < 260:
        raise ValueError("डेटा कम मिला")
    return rows


def fetch_news(query):
    url = "https://news.google.com/rss/search?q=%s&hl=en-IN&gl=IN&ceid=IN:en" % urllib.parse.quote(query + " when:2d")
    root = ET.fromstring(http_get(url))
    return [(it.findtext("title") or "").strip() for it in root.iter("item")][:8]


def sma(v, n):
    return sum(v[-n:]) / n if len(v) >= n else None


def rsi(c, n=14):
    g = l = 0.0
    for i in range(len(c) - n, len(c)):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        l += max(-d, 0)
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def core(hi, lo, cl):
    """Indicators, score and bias using data up to the last element."""
    last = cl[-1]
    ma = {n: sma(cl, n) for n in (20, 30, 50, 100, 200)}
    r14 = rsi(cl)
    trs = [max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1])) for i in range(len(cl) - 14, len(cl))]
    atr = sum(trs) / len(trs)
    score = 0
    for n in (20, 50, 200):
        if ma[n]:
            score += 1 if last > ma[n] else -1
    if ma[20] and ma[50]:
        score += 1 if ma[20] > ma[50] else -1
    score += 1 if r14 > 55 else (-1 if r14 < 45 else 0)
    bias = "BUY" if score >= 2 else ("SELL" if score <= -2 else "WAIT")
    return last, ma, r14, atr, score, bias


def day(t):
    return time.strftime("%d %b %Y", time.gmtime(t + 19800))


def stats(tr):
    if not tr:
        return None
    w = [t["p"] for t in tr if t["p"] > 0]
    l = [t["p"] for t in tr if t["p"] <= 0]
    cum = peak = dd = 0.0
    for t in tr:
        cum += t["p"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)

    def part(x):
        if not x:
            return None
        return {"n": len(x), "win": round(100 * sum(1 for t in x if t["p"] > 0) / len(x)), "tot": round(sum(t["p"] for t in x), 1)}

    k = max(1, int(len(tr) * 0.7))
    return {"n": len(tr), "win": round(100 * len(w) / len(tr)), "avg_w": round(sum(w) / len(w), 2) if w else 0,
            "avg_l": round(sum(l) / len(l), 2) if l else 0, "tot": round(cum, 1), "dd": round(dd, 1),
            "pf": round(sum(w) / abs(sum(l)), 2) if l and sum(l) != 0 else None, "a": part(tr[:k]), "b": part(tr[k:]),
            "last": [{"d": t["d"], "s": t["s"], "p": round(t["p"], 2)} for t in tr[-8:][::-1]]}


def backtest(rows):
    """Same rules as the live signal: enter on breakout of the signal day's high/low, SL 1.5 ATR, target 2.25 ATR, max 10 days."""
    hi = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    cl = [r[2] for r in rows]
    op = [r[3] for r in rows]
    tr = []
    i = 200
    while i < len(cl) - 2:
        last, ma, r14, atr, score, bias = core(hi[:i + 1], lo[:i + 1], cl[:i + 1])
        if bias == "WAIT":
            i += 1
            continue
        s = 1 if bias == "BUY" else -1
        trig = hi[i] + 0.1 * atr if s == 1 else lo[i] - 0.1 * atr
        k = i + 1
        if (s == 1 and hi[k] < trig) or (s == -1 and lo[k] > trig):
            i += 1
            continue
        entry = max(trig, op[k]) if s == 1 else min(trig, op[k])
        risk = 1.5 * atr
        sl, tp = entry - s * risk, entry + s * risk * 1.5
        ex = None
        j = k
        for j in range(k, min(k + 10, len(cl))):
            if (lo[j] <= sl) if s == 1 else (hi[j] >= sl):
                ex = sl
                break
            if (hi[j] >= tp) if s == 1 else (lo[j] <= tp):
                ex = tp
                break
        if ex is None:
            ex = cl[j]
        tr.append({"d": day(rows[k][4]), "s": bias, "p": (ex - entry) * s / entry * 100})
        i = j + 1
    return stats(tr)


def analyze(rows, step=0):
    hi = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    cl = [r[2] for r in rows]
    last, ma, r14, atr, score, bias = core(hi, lo, cl)
    p = (hi[-1] + lo[-1] + last) / 3
    piv = {"R2": p + hi[-1] - lo[-1], "R1": 2 * p - lo[-1], "P": p, "S1": 2 * p - hi[-1], "S2": p - (hi[-1] - lo[-1])}
    sh, sl_ = max(hi[-90:]), min(lo[-90:])
    fib = {str(r): sh - (sh - sl_) * r for r in (0.236, 0.382, 0.5, 0.618, 0.786)}
    buy_t, sell_t = hi[-1] + 0.1 * atr, lo[-1] - 0.1 * atr
    f = lambda x: "{:,.2f}".format(x)
    plan, when = None, []
    zone = (ma[20] - 0.25 * atr, ma[20] + 0.25 * atr)
    if bias != "WAIT":
        s = 1 if bias == "BUY" else -1
        entry = buy_t if s == 1 else sell_t
        risk = 1.5 * atr
        plan = {"entry": entry, "sl": entry - s * risk, "t1": entry + s * risk * 1.5, "t2": entry + s * risk * 2.5}
        if s == 1:
            when = ["ब्रेकआउट एंट्री: कीमत %s के ऊपर जाकर टिके (पिछले सत्र का हाई पार) तब खरीदें।" % f(buy_t),
                    "पुलबैक एंट्री: गिरावट में %s से %s के बीच रुकान दिखे तब खरीदें।" % (f(zone[0]), f(zone[1])),
                    "ट्रेड रद्द: %s के नीचे गया तो निकल जाएँ।" % f(plan["sl"])]
        else:
            when = ["ब्रेकडाउन एंट्री: कीमत %s के नीचे जाकर टिके (पिछले सत्र का लो टूटे) तब बेचें।" % f(sell_t),
                    "बाउंस एंट्री: उछाल में %s से %s के बीच कमज़ोरी दिखे तब बेचें।" % (f(zone[0]), f(zone[1])),
                    "ट्रेड रद्द: %s के ऊपर गया तो निकल जाएँ।" % f(plan["sl"])]
    else:
        when = ["अभी साफ़ संकेत नहीं है। खरीदारी तभी जब कीमत %s के ऊपर टिके, बिकवाली तभी जब %s के नीचे टिके।" % (f(buy_t), f(sell_t))]
    if r14 > 70:
        when.append("RSI %.0f: बाज़ार बहुत गर्म है, नई खरीदारी में सावधानी रखें।" % r14)
    elif r14 < 30:
        when.append("RSI %.0f: बाज़ार बहुत ठंडा है, नई बिकवाली में सावधानी रखें।" % r14)
    when.append("शुरुआती 15-30 मिनट की उछल-कूद में जल्दबाज़ी न करें। ये लेवल दिन के हिसाब से हैं, कैंडल बंद होने की पुष्टि का इंतज़ार करें।")
    strike = None
    if step and bias != "WAIT":
        strike = "%d %s" % (round(last / step) * step, "CE" if bias == "BUY" else "PE")
    rd = lambda d: {k: (round(v, 2) if v is not None else None) for k, v in d.items()}
    return {"last": round(last, 2), "date": day(rows[-1][4]), "ma": rd({str(k): v for k, v in ma.items()}),
            "rsi": round(r14, 1), "atr": round(atr, 2), "pivots": rd(piv), "fib": rd(fib), "score": score, "bias": bias,
            "plan": rd(plan) if plan else None, "when": when, "strike": strike, "bt": backtest(rows)}


def gem(prompt):
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return "AI बंद है: GOOGLE_API_KEY नहीं मिली।"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    err = ""
    for m in MODELS:
        url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % m
        try:
            out = json.loads(http_get(url, {"Content-Type": "application/json", "x-goog-api-key": key}, body, 60))
            return out["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode())["error"]["message"]
            except Exception:
                msg = ""
            err = "HTTP %s %s" % (e.code, msg[:140])
            if e.code not in (404, 429):
                break
        except Exception as e:
            err = type(e).__name__
            break
    return "AI जवाब नहीं मिला: %s" % err


RULES = "सादी हिंदी, छोटी लाइनें। नए लेवल मत बनाओ, सिर्फ़ दिए गए लेवल इस्तेमाल करो। मुनाफ़े की गारंटी मत दो।"


def brief(a):
    return json.dumps({k: a[k] for k in ("last", "ma", "rsi", "atr", "pivots", "fib", "bias", "score", "plan")}, ensure_ascii=False)


def ask_ai(name, a, news):
    return gem("तुम भारतीय बाज़ार के विश्लेषक हो। %s पर 5 लाइनों में बताओ: ख़बरों का असर, ट्रेंड, मुख्य जोखिम। %s\n\nडेटा: %s\n\nख़बरें:\n- %s"
               % (name, RULES, brief(a), "\n- ".join(news) or "कोई ख़बर नहीं"))


def team(name, a, news):
    d = brief(a)
    n = "\n- ".join(news) or "कोई ख़बर नहीं"
    with ThreadPoolExecutor(2) as ex:
        t = ex.submit(gem, "तुम टेक्निकल एनालिस्ट हो। %s के इंडिकेटर पर 4 लाइनें। %s\nडेटा: %s" % (name, RULES, d))
        w = ex.submit(gem, "तुम न्यूज़ एनालिस्ट हो। %s पर इन ख़बरों का असर 4 लाइनों में। %s\nख़बरें:\n- %s" % (name, RULES, n))
        tech, nws = t.result(), w.result()
    deb = gem("तुम दो वकील हो। टेक्निकल: %s\nन्यूज़: %s\nपहले 'तेज़ी का पक्ष' 3 लाइन, फिर 'मंदी का पक्ष' 3 लाइन। %s" % (tech, nws, RULES))
    fin = gem("तुम रिस्क मैनेजर और मुख्य ट्रेडर हो। बहस: %s\nदिया गया प्लान: %s\nसंकेत: %s\n5 लाइनों में फ़ैसला दो: ट्रेड लेना है या नहीं, जोखिम क्या है, किन हालात में ट्रेड छोड़ दें। %s"
              % (deb, json.dumps(a.get("plan")), a["bias"], RULES))
    return [["टेक्निकल एनालिस्ट", tech], ["न्यूज़ एनालिस्ट", nws], ["तेज़ी बनाम मंदी की बहस", deb], ["रिस्क मैनेजर और ट्रेडर", fin]]


def run(k, deep=False):
    if k not in INSTR:
        return {"error": "अज्ञात इंस्ट्रूमेंट"}
    name, sym, nq, step = INSTR[k]
    try:
        a = analyze(fetch_prices(sym), step)
    except Exception as e:
        return {"error": "डेटा नहीं मिला: %s। नेट चालू है?" % type(e).__name__}
    try:
        news = fetch_news(nq)
    except Exception:
        news = []
    a.update({"name": name, "news": news, "ai": "" if deep else ask_ai(name, a, news),
              "team": team(name, a, news) if deep else None,
              "note": "सोना, चाँदी, क्रूड का भाव विदेशी (USD) है, MCX का नहीं। ऑप्शन प्रीमियम यहाँ नहीं दिखता।" if not step else ""})
    return a


PAGE = r"""<!doctype html><html lang="hi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#0a1226">
<title>Levels</title><style>
:root{--tx:#eef2fb;--mut:#9aa9c7;--up:#2fd39a;--dn:#ff6b7a;--am:#f6c453;--line:rgba(255,255,255,.09)}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--tx);font:16px/1.55 system-ui,"Noto Sans Devanagari",sans-serif;padding:18px 14px 40px;
background:radial-gradient(900px 500px at 15% -10%,#26386b 0%,#101b36 55%,#0a1226 100%) fixed}
.wrap{max-width:560px;margin:auto}
h1{font-size:28px;margin:2px 0 2px;letter-spacing:.3px;background:linear-gradient(90deg,#fff,#f6c453);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);margin:0 0 14px;font-size:14px}
.tabs{display:flex;gap:8px;overflow-x:auto;padding:2px 0 10px;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}
.tab{flex:none;background:rgba(255,255,255,.06);color:var(--tx);border:1px solid var(--line);border-radius:999px;padding:9px 16px;font:inherit;font-weight:600}
.tab.on{background:linear-gradient(135deg,#f6c453,#f59e0b);color:#1b1405;border-color:transparent}
.tab:focus-visible,summary:focus-visible{outline:2px solid var(--am)}
.tm{display:flex;align-items:center;gap:8px;color:var(--mut);font-size:14px;margin:2px 0 12px}
.card{background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:18px;padding:16px;margin-top:12px;backdrop-filter:blur(8px)}
.card h3,summary{margin:0 0 10px;font-size:15px;color:var(--mut);font-weight:600}
details.card summary{margin:0;cursor:pointer}details[open] summary{margin-bottom:10px}
.hero.BUY{background:linear-gradient(135deg,rgba(47,211,154,.2),rgba(255,255,255,.04))}
.hero.SELL{background:linear-gradient(135deg,rgba(255,107,122,.2),rgba(255,255,255,.04))}
.hero.WAIT{background:linear-gradient(135deg,rgba(246,196,83,.16),rgba(255,255,255,.04))}
.big{font-size:40px;font-weight:700;line-height:1.15;font-variant-numeric:tabular-nums;margin:2px 0 8px}
.pill{display:inline-block;padding:7px 16px;border-radius:999px;font-weight:700;font-size:18px}
.pill.BUY{background:var(--up);color:#052a1d}.pill.SELL{background:var(--dn);color:#32060d}.pill.WAIT{background:var(--am);color:#2b2005}
.meter{height:8px;border-radius:99px;background:linear-gradient(90deg,var(--dn),var(--am),var(--up));position:relative;margin:16px 4px 6px}
.mk{position:absolute;top:-5px;width:5px;height:18px;background:#fff;border-radius:3px;box-shadow:0 0 0 2px rgba(0,0,0,.35)}
.lad{display:flex;justify-content:space-between;padding:11px 12px;margin:6px 0;border-radius:10px;border-left:5px solid var(--mut);background:rgba(255,255,255,.05);font-variant-numeric:tabular-nums}
.lad.t{border-color:var(--up)}.lad.e{border-color:var(--am)}.lad.s{border-color:var(--dn)}
.r{display:flex;justify-content:space-between;padding:4px 0;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.dn{color:var(--dn)}
.n{white-space:pre-wrap;margin:7px 0;font-size:15px}.mut{color:var(--mut);font-size:13px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px}
.stat{background:rgba(255,255,255,.05);border-radius:12px;padding:10px 6px;text-align:center}.stat b{display:block;font-size:18px}.stat span{font-size:12px;color:var(--mut)}
.skel{height:110px;border-radius:18px;margin-top:12px;background:linear-gradient(90deg,rgba(255,255,255,.05) 25%,rgba(255,255,255,.12) 50%,rgba(255,255,255,.05) 75%);background-size:200% 100%;animation:sh 1.3s infinite}
@keyframes sh{to{background-position:-200% 0}}
</style></head><body><div class="wrap"><h1>Levels</h1><p class="sub">डेटा, ख़बरें, एंट्री का समय और पुरानी जाँच, एक जगह</p>
<div class="tabs" id="b"></div>
<label class="tm"><input type="checkbox" id="dp"> टीम मोड: 4 AI एजेंट, धीमा</label><div id="o"></div>
<p class="mut">यह सिर्फ़ जानकारी है, सलाह नहीं। हमेशा स्टॉप-लॉस लगाएँ।</p></div><script>
const K=[["NIFTY","Nifty"],["SENSEX","Sensex"],["BANKNIFTY","Bank Nifty"],["GOLD","सोना"],["SILVER","चाँदी"],["CRUDE","क्रूड"]];
let cur="NIFTY";const $=s=>document.getElementById(s);
const fmt=x=>x===null||x===undefined?"-":Number(x).toLocaleString("en-IN",{maximumFractionDigits:2});
function E(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
function card(c){return E("div","card "+(c||""))}
function det(t){const d=E("details","card");d.append(E("summary","",t));return d}
function rows(s,o){for(const k in o){const r=E("div","r");r.append(E("span","",k),E("b","",fmt(o[k])));s.append(r)}}
function sk(){const d=E("div");for(let i=0;i<3;i++)d.append(E("div","skel"));return d}
const API=()=>fetch("/api/analyze?k="+cur+($("dp").checked?"&deep=1":""),{headers:{"X-Pin":localStorage.getItem("pin")||""}}).then(r=>r.json());
K.forEach(([k,n])=>{const b=E("button","tab"+(k===cur?" on":""),n);b.onclick=()=>{cur=k;document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));b.classList.add("on");go()};$("b").append(b)});
$("dp").onchange=go;
async function go(){const o=$("o");o.replaceChildren(sk());
try{let d=await API();
if(d.error==="PIN"){const p=prompt("PIN डालें");if(!p){o.replaceChildren(E("div","card","PIN चाहिए"));return}localStorage.setItem("pin",p);d=await API();
if(d.error==="PIN"){localStorage.removeItem("pin");o.replaceChildren(E("div","card","❌ PIN गलत है"));return}}
if(d.error){o.replaceChildren(E("div","card","❌ "+d.error));return}
o.replaceChildren(...build(d))}catch(e){o.replaceChildren(E("div","card","❌ जवाब नहीं आया। फ्री सर्वर जाग रहा होगा, 1 मिनट बाद फिर कोशिश करें।"))}}
function build(d){const out=[];
const lab={BUY:"खरीदें · Call",SELL:"बेचें · Put",WAIT:"अभी रुकें"}[d.bias];
let h=card("hero "+d.bias);h.append(E("div","mut",d.name+" · "+d.date+" का बंद भाव"),E("div","big",fmt(d.last)),E("div","pill "+d.bias,lab));
if(d.strike)h.append(E("div","mut","स्ट्राइक का अंदाज़ा: "+d.strike));
const m=E("div","meter"),mk=E("i","mk");mk.style.left="calc("+((d.score+5)/10*100)+"% - 2px)";m.append(mk);
h.append(m,E("div","mut","संकेत की ताक़त: "+d.score+" (−5 बहुत मंदी, +5 बहुत तेज़ी)"));out.push(h);
if(d.plan){const c=card();c.append(E("h3","","एंट्री, स्टॉप-लॉस, टारगेट"));
[["टारगेट 2",d.plan.t2,"t"],["टारगेट 1",d.plan.t1,"t"],["एंट्री",d.plan.entry,"e"],["स्टॉप-लॉस",d.plan.sl,"s"]].sort((a,b)=>b[1]-a[1]).forEach(([a,v,k])=>{const r=E("div","lad "+k);r.append(E("span","",a),E("b","",fmt(v)));c.append(r)});out.push(c)}
const w=card();w.append(E("h3","","कब एंट्री लें"));d.when.forEach(t=>w.append(E("div","n","• "+t)));out.push(w);
const c=card();c.append(E("h3","","पुराने 5 साल के डेटा पर इसी नियम की जाँच"));
if(d.bt){const b=d.bt,g=E("div","grid3");
[["ट्रेड",b.n],["जीत",b.win+"%"],["कुल",(b.tot>0?"+":"")+b.tot+"%"],["सबसे बड़ी गिरावट",b.dd+"%"],["प्रॉफिट फैक्टर",b.pf===null?"-":b.pf],["औसत जीत/हार",b.avg_w+"/"+b.avg_l+"%"]].forEach(([a,v])=>{const x=E("div","stat");x.append(E("b","",v),E("span","",a));g.append(x)});c.append(g);
if(b.a&&b.b){c.append(E("div","n","पहले 70% ट्रेड: जीत "+b.a.win+"%, कुल "+b.a.tot+"%"),E("div","n","बाद के 30% (अनदेखा हिस्सा): जीत "+b.b.win+"%, कुल "+b.b.tot+"%"))}
b.last.forEach(t=>{const r=E("div","r");r.append(E("span","",t.d+" · "+(t.s==="BUY"?"खरीद":"बिक्री")),E("b",t.p>=0?"up":"dn",(t.p>=0?"+":"")+t.p+"%"));c.append(r)})}
else c.append(E("div","n","जाँच के लिए पर्याप्त ट्रेड नहीं बने।"));
c.append(E("p","mut","यह कीमत की चाल पर है। ऑप्शन प्रीमियम, ब्रोकरेज और स्लिपेज शामिल नहीं। पुराना नतीजा भविष्य की गारंटी नहीं।"));out.push(c);
if(d.team){d.team.forEach(([t,x])=>{const s=card();s.append(E("h3","",t),E("div","n",x));out.push(s)})}
else{const s=card();s.append(E("h3","","AI की राय"),E("div","n",d.ai));out.push(s)}
let s=det("इंडिकेटर");rows(s,d.ma);rows(s,{"RSI (14)":d.rsi,"ATR":d.atr});out.push(s);
s=det("सपोर्ट और रेज़िस्टेंस (पिवट)");rows(s,d.pivots);out.push(s);
s=det("Fibonacci (90 दिन)");rows(s,d.fib);out.push(s);
s=det("ताज़ा ख़बरें");if(!d.news.length)s.append(E("div","n","कोई ख़बर नहीं मिली"));d.news.forEach(t=>s.append(E("div","n","• "+t)));out.push(s);
if(d.note)out.push(E("p","mut",d.note));return out}
go();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/analyze":
            q = urllib.parse.parse_qs(u.query)
            ok = not PIN or hmac.compare_digest(self.headers.get("X-Pin", ""), PIN)
            out = run(q.get("k", [""])[0], q.get("deep", [""])[0] == "1") if ok else {"error": "PIN"}
            body, ct = json.dumps(out, ensure_ascii=False).encode(), "application/json; charset=utf-8"
        elif u.path == "/":
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    if HOST != "127.0.0.1" and not PIN:
        raise SystemExit("सार्वजनिक सर्वर पर APP_PIN ज़रूरी है, वरना कोई भी आपकी AI key खर्च कर देगा।")
    print("चालू है। ब्राउज़र में खोलें: http://127.0.0.1:%d  (बंद करने के लिए CTRL+c)" % PORT)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
