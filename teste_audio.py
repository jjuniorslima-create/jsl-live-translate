"""
Teste automático do JSL Live Translate.

Manda uma gravação real de voz para o servidor, como se fosse o telemóvel,
e mede o que volta: quantidade de som, duração implícita e tempo real.
Serve para descobrir se o som é devolvido à velocidade certa.
"""
import asyncio
import json
import sys
import time
import wave

import websockets

WAV = sys.argv[1] if len(sys.argv) > 1 else "gravacoes/traducao_fr_20260630_235751.wav"
LANG_ALVO = sys.argv[2] if len(sys.argv) > 2 else "pt"
URL = "ws://localhost:8000/ws"

TAXA_ENVIO = 16000
BLOCO_MS = 100


def ler_wav_para_16k(caminho):
    w = wave.open(caminho, "rb")
    taxa = w.getframerate()
    canais = w.getnchannels()
    dados = w.readframes(w.getnframes())
    w.close()

    import array
    amostras = array.array("h")
    amostras.frombytes(dados)
    if canais == 2:
        amostras = array.array("h", amostras[0::2])

    if taxa == TAXA_ENVIO:
        return amostras.tobytes(), taxa

    # Reamostragem por média de blocos (mesmo método do browser)
    r = taxa / TAXA_ENVIO
    n = int(len(amostras) / r)
    saida = array.array("h", [0] * n)
    for i in range(n):
        a = int(i * r)
        b = min(int((i + 1) * r), len(amostras))
        if b > a:
            saida[i] = sum(amostras[a:b]) // (b - a)
    return saida.tobytes(), taxa


async def main():
    pcm, taxa_original = ler_wav_para_16k(WAV)
    dur_entrada = len(pcm) / 2 / TAXA_ENVIO
    print(f"Entrada: {WAV}")
    print(f"  taxa original {taxa_original} Hz -> enviada a {TAXA_ENVIO} Hz")
    print(f"  duracao da fala: {dur_entrada:.1f} s")
    print(f"  idioma alvo: {LANG_ALVO}")
    print("-" * 60)

    bytes_bloco = int(TAXA_ENVIO * 2 * BLOCO_MS / 1000)
    recebido = bytearray()
    blocos = []          # (bytes, pico) por bocado recebido
    transcricoes = []
    t_inicio = None
    t_primeiro_audio = None

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "lang": LANG_ALVO}))

        async def receber():
            nonlocal t_primeiro_audio
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        if t_primeiro_audio is None and t_inicio:
                            t_primeiro_audio = time.time() - t_inicio
                        recebido.extend(msg)
                        import array as _a
                        _s = _a.array("h"); _s.frombytes(msg)
                        pico = max(abs(x) for x in _s) if _s else 0
                        blocos.append((len(msg), pico))
                    else:
                        m = json.loads(msg)
                        if m.get("type", "").startswith("transcript"):
                            transcricoes.append(f"  {m['type']}: {m.get('text','')}")
                        elif m.get("type") == "error":
                            print("  ERRO:", m.get("msg"))
            except Exception:
                pass

        tarefa = asyncio.create_task(receber())
        await asyncio.sleep(1.0)  # deixa ligar ao Gemini

        t_inicio = time.time()
        for i in range(0, len(pcm), bytes_bloco):
            await ws.send(pcm[i:i + bytes_bloco])
            await asyncio.sleep(BLOCO_MS / 1000)  # ritmo real
        t_fim_envio = time.time() - t_inicio
        print(f"Fala enviada em {t_fim_envio:.1f} s (ritmo real)")

        # Espera pela tradução
        await asyncio.sleep(12)
        await ws.send(json.dumps({"type": "stop"}))
        tarefa.cancel()

    print("-" * 60)
    for t in transcricoes[-8:]:
        print(t)
    print("-" * 60)

    n = len(recebido)
    print(f"Som devolvido: {n} bytes ({n//2} amostras)")
    if n:
        print(f"  1o som chegou {t_primeiro_audio:.1f} s depois de comecar a falar")
        for taxa in (16000, 24000, 48000):
            print(f"  se for {taxa} Hz -> {n/2/taxa:.1f} s de audio")
        print()
        print(f"  A fala original tinha {dur_entrada:.1f} s.")
        print()
        LIMIAR = 300
        silencio = sum(b for b, p in blocos if p < LIMIAR)
        fala = sum(b for b, p in blocos if p >= LIMIAR)
        print(f"  Bocados recebidos: {len(blocos)}")
        print(f"  SILENCIO (pico < {LIMIAR}): {silencio} bytes = {silencio/2/24000:.1f} s"
              f"  ({100*silencio/max(1,n):.0f}%)")
        print(f"  FALA     (pico >= {LIMIAR}): {fala} bytes = {fala/2/24000:.1f} s"
              f"  ({100*fala/max(1,n):.0f}%)")
        print()
        print(f"  >> Com o silencio saltado, a fila teria {fala/2/24000:.1f} s"
              f" em vez de {n/2/24000:.1f} s.")
    else:
        print("  NENHUM som devolvido.")


asyncio.run(main())
