#!/usr/bin/env python3
"""Levels: one-tap data, indicators, news and trade levels. Pure Python, no pip installs.
Run:  python app.py   then open  http://127.0.0.1:8080  in the phone browser."""
import hmac, json, os, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
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
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%s?range=1y&interval=1d" % urllib.parse.quote(sym)
    q = json.loads(http_get(url))["chart"]["result"][0]["indicators"]["quote"][0]
    rows = [(h, l, c) for h, l, c in zip(q["high"], q["low"], q["close"]) if None not in (h, l, c)]
    if len(rows) < 30:
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


def analyze(rows, step=0):
    hi = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    cl = [r[2] for r in rows]
    last = cl[-1]
    ma = {n: sma(cl, n) for n in (20, 30, 50, 100, 200)}
    r14 = rsi(cl)
    trs = [max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1])) for i in range(len(cl) - 14, len(cl))]
    atr = sum(trs) / len(trs)
    p = (hi[-1] + lo[-1] + last) / 3
    piv = {"R2": p + hi[-1] - lo[-1], "R1": 2 * p - lo[-1], "P": p, "S1": 2 * p - hi[-1], "S2": p - (hi[-1] - lo[-1])}
    sh, sl = max(hi[-90:]), min(lo[-90:])
    fib = {str(r): sh - (sh - sl) * r for r in (0.236, 0.382, 0.5, 0.618, 0.786)}
    score = 0
    for n in (20, 50, 200):
        if ma[n]:
            score += 1 if last > ma[n] else -1
    if ma[20] and ma[50]:
        score += 1 if ma[20] > ma[50] else -1
    score += 1 if r14 > 55 else (-1 if r14 < 45 else 0)
    bias = "BUY" if score >= 2 else ("SELL" if score <= -2 else "WAIT")
    plan = None
    if bias != "WAIT":
        s = 1 if bias == "BUY" else -1
        risk = 1.5 * atr
        plan = {"entry": last, "sl": last - s * risk, "t1": last + s * risk * 1.5, "t2": last + s * risk * 2.5}
    strike = None
    if step and bias != "WAIT":
        strike = "%d %s" % (round(last / step) * step, "CE" if bias == "BUY" else "PE")
    rd = lambda d: {k: (round(v, 2) if v is not None else None) for k, v in d.items()}
    return {"last": round(last, 2), "ma": rd({str(k): v for k, v in ma.items()}), "rsi": round(r14, 1),
            "atr": round(atr, 2), "pivots": rd(piv), "fib": rd(fib), "score": score, "bias": bias,
            "plan": rd(plan) if plan else None, "strike": strike}


def gem(prompt):
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return "AI बंद है: GOOGLE_API_KEY नहीं मिली।"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % MODEL
    try:
        out = json.loads(http_get(url, {"Content-Type": "application/json", "x-goog-api-key": key}, body, 60))
        return out["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return "AI जवाब नहीं मिला (%s)। GEMINI_MODEL या key जाँचें।" % type(e).__name__


RULES = "सादी हिंदी, छोटी लाइनें। नए लेवल मत बनाओ, सिर्फ़ दिए गए लेवल इस्तेमाल करो। मुनाफ़े की गारंटी मत दो।"


def ask_ai(name, a, news):
    return gem("तुम भारतीय बाज़ार के विश्लेषक हो। %s पर 5 लाइनों में बताओ: ख़बरों का असर, ट्रेंड, मुख्य जोखिम। %s\n\nडेटा: %s\n\nख़बरें:\n- %s"
               % (name, RULES, json.dumps(a, ensure_ascii=False), "\n- ".join(news) or "कोई ख़बर नहीं"))


def team(name, a, news):
    d = json.dumps(a, ensure_ascii=False)
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


PAGE = """<!doctype html><html lang="hi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#14213d">
<title>Levels</title><style>
:root{--bg:#14213d;--card:#1d2f52;--tx:#eef2f9;--mut:#9fb0cc;--up:#3ccf91;--dn:#ff6b6b;--am:#f4b942}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:16px/1.5 system-ui,"Noto Sans Devanagari",sans-serif;padding:16px;max-width:560px;margin:auto}
h1{font-size:22px;margin:4px 0 14px}.btns{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
button{background:var(--card);color:var(--tx);border:1px solid #33497a;border-radius:10px;padding:14px 4px;font:inherit;font-weight:600}
button:active,button:focus-visible{outline:2px solid var(--am)}
.sec{background:var(--card);border-radius:12px;padding:14px;margin-top:12px}.sec h3{margin:0 0 8px;font-size:16px;color:var(--mut);font-weight:600}
.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}.BUY{color:var(--up)}.SELL{color:var(--dn)}.WAIT{color:var(--am)}
.lad{display:flex;justify-content:space-between;padding:9px 10px;margin:4px 0;border-left:5px solid var(--mut);background:#16294a;font-variant-numeric:tabular-nums}
.lad.t{border-color:var(--up)}.lad.e{border-color:var(--am)}.lad.s{border-color:var(--dn)}
.r{display:flex;justify-content:space-between;padding:3px 0;font-variant-numeric:tabular-nums}
.n{white-space:pre-wrap;margin:6px 0;color:var(--tx);font-size:14px}.mut{color:var(--mut);font-size:13px}
</style></head><body><h1>Levels</h1><div class="btns" id="b"></div><label class="mut"><input type="checkbox" id="dp"> टीम मोड: 4 AI एजेंट, धीमा</label><div id="o"></div>
<p class="mut">यह सिर्फ़ जानकारी है, सलाह नहीं। हमेशा स्टॉप-लॉस लगाएँ।</p><script>
const K=[["NIFTY","Nifty"],["SENSEX","Sensex"],["BANKNIFTY","Bank Nifty"],["GOLD","सोना"],["SILVER","चाँदी"],["CRUDE","क्रूड"]];
const $=s=>document.getElementById(s);
function E(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
function sec(t){const s=E("div","sec");s.append(E("h3","",t));return s}
function rows(s,o){for(const k in o){const r=E("div","r");r.append(E("span","",k),E("b","",o[k]===null?"-":o[k]));s.append(r)}}
K.forEach(([k,n])=>{const b=E("button","",n);b.onclick=()=>go(k,n);$("b").append(b)});
async function go(k,n){const o=$("o");o.textContent="⏳ "+n+" का विश्लेषण हो रहा है...";
try{const api=()=>fetch("/api/analyze?k="+k+($("dp").checked?"&deep=1":""),{headers:{"X-Pin":localStorage.getItem("pin")||""}}).then(r=>r.json());
let d=await api();
if(d.error==="PIN"){const p=prompt("PIN डालें");if(!p){o.textContent="PIN चाहिए";return}localStorage.setItem("pin",p);d=await api();
if(d.error==="PIN"){localStorage.removeItem("pin");o.textContent="❌ PIN गलत है";return}}
o.textContent="";
if(d.error){o.textContent="❌ "+d.error;return}
let s=sec(d.name);s.append(E("div","big",d.last));
const lab={BUY:"खरीदें (Call)",SELL:"बेचें (Put)",WAIT:"रुकें, साफ़ संकेत नहीं"}[d.bias];
s.append(E("div","big "+d.bias,lab));if(d.strike)s.append(E("div","mut","स्ट्राइक का अंदाज़ा: "+d.strike));o.append(s);
if(d.plan){s=sec("एंट्री, स्टॉप-लॉस, टारगेट");const L=[["टारगेट 2",d.plan.t2,"t"],["टारगेट 1",d.plan.t1,"t"],["एंट्री",d.plan.entry,"e"],["स्टॉप-लॉस",d.plan.sl,"s"]].sort((a,b)=>b[1]-a[1]);
L.forEach(([a,v,c])=>{const r=E("div","lad "+c);r.append(E("span","",a),E("b","",v.toFixed(2)));s.append(r)});o.append(s)}
if(d.team){d.team.forEach(([t,x])=>{s=sec(t);s.append(E("div","n",x));o.append(s)})}else{s=sec("AI की राय");s.append(E("div","n",d.ai));o.append(s)}
s=sec("मूविंग एवरेज");rows(s,d.ma);rows(s,{"RSI (14)":d.rsi,"ATR":d.atr});o.append(s);
s=sec("सपोर्ट और रेज़िस्टेंस (पिवट)");rows(s,d.pivots);o.append(s);
s=sec("Fibonacci (90 दिन)");rows(s,d.fib);o.append(s);
s=sec("ताज़ा ख़बरें");if(!d.news.length)s.append(E("div","n","कोई ख़बर नहीं मिली"));d.news.forEach(t=>s.append(E("div","n","• "+t)));o.append(s);
if(d.note)o.append(E("p","mut",d.note))}catch(e){o.textContent="❌ सर्वर से जवाब नहीं आया: "+e}}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/analyze":
            k = urllib.parse.parse_qs(u.query).get("k", [""])[0]
            ok = not PIN or hmac.compare_digest(self.headers.get("X-Pin", ""), PIN)
            deep = urllib.parse.parse_qs(u.query).get("deep", [""])[0] == "1"
            out = run(k, deep) if ok else {"error": "PIN"}
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
