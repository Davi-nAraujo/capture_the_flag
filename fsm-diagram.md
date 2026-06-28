# Máquina de estados da missão (`mission_fsm`)

Diagrama fiel ao código em `src/capture_the_flag/capture_the_flag/mission_fsm.py`.
Tick em duas fases a ~10 Hz: **(1)** avalia a transição do estado atual; **(2)** executa
o comportamento do estado. `OBSTACLE_AVOIDANCE` é um *override* por subsumption: guarda
quem o chamou em `previous_state` e retorna para lá quando a frente libera.

```mermaid
stateDiagram-v2

    IDLE : IDLE — cmd_vel = 0. Espera o 1º /scan. Lê /scan
    EXPLORING : EXPLORING — anda reto (linear.x fixo). Lê /scan e /camera
    OBSTACLE_AVOIDANCE : OBSTACLE_AVOIDANCE — desvio em 3 fases (BACKUP→TURN→ESCAPE), gira p/ o lado da bandeira/livre, com histerese + anti-livelock. Lê /scan + previous_state
    FLAG_FOUND_CONFIRMATION : FLAG_FOUND_CONFIRMATION — cmd_vel = 0. Observa ~2 s p/ filtrar falso-positivo. Lê /camera
    GOING_TO_FLAG : GOING_TO_FLAG — go-to-point até a META lembrada (frame odom) + repulsão lateral. Lê /scan, /camera, /odom_gt
    REFINDING_FLAG : REFINDING_FLAG — gira no lugar p/ o último lado visto. Lê /camera
    ADJUSTING_POSITION_TO_COLLECT_FLAG : ADJUSTING_POSITION — linear.x = 0, gira p/ centralizar a bandeira. Lê /camera
    COLLECTING_FLAG : COLLECTING_FLAG — cmd_vel = 0 por ~5 s, loga o sucesso. Lê /camera
    RETURNING_HOME : RETURNING_HOME — estado FINAL: missão concluída, robô parado

    [*] --> IDLE

    IDLE --> EXPLORING : 1º /scan recebido

    EXPLORING --> FLAG_FOUND_CONFIRMATION : bandeira visível
    EXPLORING --> OBSTACLE_AVOIDANCE : frente bloqueada

    OBSTACLE_AVOIDANCE --> EXPLORING : frente livre (retorno)
    OBSTACLE_AVOIDANCE --> GOING_TO_FLAG : frente livre (retorno)
    OBSTACLE_AVOIDANCE --> FLAG_FOUND_CONFIRMATION : escapou + bandeira à vista

    FLAG_FOUND_CONFIRMATION --> GOING_TO_FLAG : confirmada (~2 s)
    FLAG_FOUND_CONFIRMATION --> EXPLORING : falso-positivo

    GOING_TO_FLAG --> ADJUSTING_POSITION_TO_COLLECT_FLAG : área ≥ AREA_NEAR (chegou)
    GOING_TO_FLAG --> OBSTACLE_AVOIDANCE : frente bloqueada
    GOING_TO_FLAG --> REFINDING_FLAG : sem meta estimada

    REFINDING_FLAG --> GOING_TO_FLAG : bandeira reencontrada
    REFINDING_FLAG --> EXPLORING : timeout (~12 s)

    ADJUSTING_POSITION_TO_COLLECT_FLAG --> COLLECTING_FLAG : orientação ajustada
    ADJUSTING_POSITION_TO_COLLECT_FLAG --> REFINDING_FLAG : bandeira perdida

    COLLECTING_FLAG --> RETURNING_HOME : bandeira coletada
    COLLECTING_FLAG --> REFINDING_FLAG : bandeira perdida

    RETURNING_HOME --> [*] : estado final (parado)
```

---

## Extensão para o Trabalho 2 — A PROJETAR (tarefa do aluno, critério #1 "expandir a máquina de estados")

O grafo acima é o congelado do T1. O T2 exige *expandir* esta máquina. **Você desenha** — abaixo
só as perguntas de projeto a responder (não a resposta):

1. **Captura real:** o `COLLECTING_FLAG` deixa de ser simbólico e precisa acionar a garra. Isso
   vira **sub-fases dentro de um estado** (como o `OBSTACLE_AVOIDANCE` tem BACKUP→TURN→ESCAPE) ou
   **estados novos** (ex.: `APPROACH_GRASP`→`CLOSING_GRIPPER`→`LIFTING`)? Trade-off? (Uma tentativa
   em 2026-06-28 foi REVERTIDA — reconstruir SOBRE a chegada por área do T1, não no lugar dela.)
2. **Retorno:** `RETURNING_HOME` deixa de ser terminal e vira navegação go-to-point até a
   **pose inicial registrada**. Qual a guarda de "cheguei em casa"? E se a frente bloquear no
   caminho (a subsumption do desvio ainda vale carregando a bandeira)?
3. **Depósito:** estado novo `DEPOSITING_FLAG`. Sequência (posicionar → abrir garra → recuar) —
   estado único com fases ou vários? Guarda de "depositado"? Localiza por odom ou câmera?
4. **Busca dirigida:** o `EXPLORING` vira *traverse* dirigido à zona azul (a bandeira é fixa).
   Muda o grafo ou só o comportamento do estado existente?
5. **Quais arestas novas** aparecem e **quais guardas** as disparam? Desenhe o grafo T2 completo
   aqui quando decidir, do mesmo jeito (com os contratos de comportamento inline).
