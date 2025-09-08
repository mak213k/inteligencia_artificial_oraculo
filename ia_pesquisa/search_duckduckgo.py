from agno.agent import Agent

from agno.models.huggingface import HuggingFace


from agno.tools.duckduckgo import DuckDuckGoTools
from agno.memory.v2.memory import Memory

import os
os.environ["HF_TOKEN"] = ""

def criar_agente_info(stream_response=True):
    memoria = Memory()

    agente = Agent(
        name="Info Agent",
        role="Assistente para buscar e fornecer informações detalhadas e confiáveis",
        model=HuggingFace(
        id="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=4096, temperature=0
        ),
        tools=[DuckDuckGoTools()],
        instructions=(
            "Forneça respostas detalhadas, claras e baseadas em fontes confiáveis. "
            "Use a ferramenta DuckDuckGo para obter dados atualizados. "
            "Lembre-se do contexto das interações anteriores."
        ),
        memory=memoria,
        show_tool_calls=True,
        markdown=True,
        stream=stream_response,
    )
    return agente

def interagir_com_agente(agente):
    print("Pergunte algo para o agente (digite 'sair' para encerrar):")
    while True:
        try:
            pergunta = input("\nVocê: ").strip()
            if pergunta.lower() in {"sair", "exit", "quit"}:
                print("Encerrando a interação. Até mais!")
                break

            agente.print_response(pergunta, stream=True)

        except KeyboardInterrupt:
            print("\nInteração interrompida.")
            break
        except Exception as e:
            print(f"Erro: {e}")
            break

if __name__ == "__main__":
    meu_agente = criar_agente_info(stream_response=True)
    interagir_com_agente(meu_agente)