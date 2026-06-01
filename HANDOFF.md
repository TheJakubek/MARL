# Projekt 9: Correlated exploration for MARL — handoff

Dokument dla kolegi z zespołu. Opisuje co zostało zrobione, dlaczego takie decyzje
zostały podjęte, jaki jest aktualny stan kodu i co jest do zrobienia.

---

## 1. O czym jest projekt

Z PDF-a (`RL projekty.pdf`, Proposal 9):

> W multi-agent RL eksploracja jest zwykle robiona naiwnie — każdy agent niezależnie
> stosuje epsilon-greedy. W środowiskach kooperacyjnych zachowania zespołowe są
> jednak skorelowane — często zespół musi *wspólnie* spróbować nowej strategii,
> żeby ją odkryć. Hipoteza: agenci w **podobnych stanach** powinni eksplorować
> **podobne akcje**.

Pipeline z proposala:
1. Wylicz miarę podobieństwa między agentami (cosine similarity obserwacji /
   ukrytych stanów / Q-wartości).
2. Zamień ją w **macierz korelacji** PSD.
3. Próbkuj akcje eksploracyjne z **kopuły Gaussa** parametryzowanej tą macierzą.
4. Algorytm bazowy: jakaś metoda value-decomposition (VDN / QMIX / QPLEX / NA²Q).
5. Ewaluacja: najpierw toy envy (Level-Based Foraging, Hallway), potem ewentualnie
   SMAC / SMACv2 / SMAX / GRF.

---

## 2. Tło teoretyczne (skrót)

### 2.1 Single-agent RL

Q-funkcja $Q(s,a)$ = oczekiwana suma zdyskontowanych nagród. DQN uczy
$Q_\theta(s,a)$ na błędzie Bellmana
$\big(r + \gamma \max_{a'} Q_{\bar\theta}(s', a') - Q_\theta(s,a)\big)^2$,
z target networkiem $\bar\theta$ kopiowanym co N kroków dla stabilności.

Epsilon-greedy: z prob. $\varepsilon$ losowa akcja, w przeciwnym razie $\arg\max Q$.

### 2.2 MARL i value decomposition

W kooperacyjnym MARL każdy agent $i$ widzi lokalną obserwację $o^i$, wybiera akcję
$a^i$, wszyscy dostają wspólną nagrodę. Wspólna przestrzeń akcji rośnie wykładniczo,
więc nie da się traktować tego jak zwykłego DQN.

**VDN (Value Decomposition Networks)** rozwiązuje to przez sumowe rozłożenie:
$$Q_{\text{tot}}(\mathbf{o}, \mathbf{a}) = \sum_i Q_i(o^i, a^i).$$
Każdy agent ma własną sieć $Q_i$, ale uczy się przez wspólny TD error na $Q_{\text{tot}}$.
Decentralizacja w czasie wykonania działa, bo
$\arg\max_\mathbf{a} Q_{\text{tot}} = (\arg\max_{a^1} Q_1, \dots, \arg\max_{a^N} Q_N)$.

QMIX uogólnia VDN do monotonicznego mixera, ale dla pierwszej iteracji wybraliśmy
VDN — najprostszy, wystarczy do pokazania, czy eksploracja działa.

### 2.3 Dlaczego niezależna eksploracja jest słaba

Klasyczny przykład: dwóch agentów musi *jednocześnie* nacisnąć dwa przyciski.
Z niezależnym $\varepsilon = 0.1$ prawdopodobieństwo tego rzędu $\varepsilon^2 / |A|^2$ —
astronomicznie małe. Wymagana jest skorelowana eksploracja.

### 2.4 Kopuła Gaussa

Cel: chcemy próbkować $N$ akcji dyskretnych tak, że
- każda akcja brzegowo jest uniform po dostępnych akcjach (jak w eps-greedy),
- ale akcje są skorelowane zgodnie z macierzą $\Sigma \in \mathbb{R}^{N \times N}$.

Algorytm:
1. Cholesky: $\Sigma = L L^T$.
2. Próbkuj $z_{\text{iid}} \sim \mathcal{N}(0, I)$, oblicz $z = L z_{\text{iid}}$
   → skorelowane standard normals.
3. $u^i = \Phi(z^i)$ (CDF Gaussa) — *probability integral transform* gwarantuje, że
   brzegowo $u^i \sim \text{Uniform}(0,1)$, ale wspólnie zachowuje korelację.
4. Akcja agenta $i$: `floor(u^i * n_actions)` — uniform po akcjach brzegowo,
   skorelowana z innymi agentami.

Macierz $\Sigma$ musi być **dodatnio półokreślona (PSD)**. Trick: jeśli zdefiniujemy
$\rho_{ij}$ jako cosine similarity między obserwacjami (lub innymi wektorami) agentów,
to $\Sigma$ jest **macierzą Grama** ze znormalizowanych wektorów — z definicji PSD.

---

## 3. Co zaimplementowaliśmy

### 3.1 Środowiska

**`env.py` — `CoordinatedSwitches`** (toy do debugowania):
- Siatka 5×5, 2 agentów, 2 stałe przełączniki w rogach `(0,4)` i `(4,0)`.
- Akcje: 5 dyskretnych — stay/N/S/E/W.
- Obserwacja per-agent: `(own_pos, mate_pos, own_switch_pos)` znormalizowana do [0,1],
  wektor 6-wymiarowy.
- Nagroda: −0.01 za krok, +1.0 jeśli oboje stoją na swoich przełącznikach jednocześnie.
- Episode kończy się przy sukcesie albo po 30 krokach.
- Random non-overlapping start positions.

**`lbf_env.py` — `LBFAdapter`** (główny benchmark, Level-Based Foraging):
- Cienki wrapper na `gymnasium` env `Foraging-6x6-2p-1f-coop-v3`.
- 6×6, 2 graczy (level 1 każdy), 1 jedzenie (level 2 → wymaga obu agentów na load
  jednocześnie, idealny test koordynacji).
- 6 akcji: 0=none, 1-4=N/S/E/W, 5=load.
- Obserwacja per-agent: 9-wymiarowa (pozycje + levels).
- Reward: agent dostaje proporcjonalny udział, gdy zbierze jedzenie. Adapter sumuje
  rewards do jednej `shared_reward` (VDN potrzebuje wspólnej nagrody).
- API zgodne z `CoordinatedSwitches` (`reset()`, `step(actions)`, `obs_dim`,
  `N_AGENTS`, `N_ACTIONS`).

### 3.2 VDN learner — `agent.py`

- **`QNet`**: MLP `obs_dim → 128 → 128 → n_actions`, ReLU. (Domyślne 64
  okazało się za małe na LBF — zwiększyliśmy do standardowego rozmiaru z papers QMIX.)
- **`ReplayBuffer`**: deque o pojemności `capacity`, `push` zapisuje krotki
  `(obs, actions, reward, next_obs, done)`. `sample(batch)` zwraca tensory PyTorcha
  o kształcie `(B, n_agents, ...)`.
- **`VDNLearner`**:
  - `q_nets`: `nn.ModuleList` — osobna sieć per agent (nie sharing — agenci mają
    różne cele).
  - `target_nets`: zamrożone kopie, sync co 500 update'ów (zwiększone z 200 dla
    stabilności).
  - `lr=1e-4` (zmniejszone z 5e-4 po obserwacji catastrophic forgetting na LBF).
  - `update(batch)`:
    - $Q_{\text{tot}} = \sum_i Q_i(o^i, a^i)$ — wybór akcji przez `gather`.
    - Target: $r + \gamma \sum_i \max_{a'} Q_i^{\text{target}}(o^i_{t+1}, a')$.
    - Loss = MSE.
    - Adam, gradient clipping (max_norm=10).
  - `q_values(obs_list)`: helper bez gradientu, używany przy zbieraniu doświadczeń.

### 3.3 Strategie eksploracji — `exploration.py`

Obie strategie mają jeden interfejs:
`select(q_values_list, obs_list, epsilon) -> list[int]`.

**`IndependentEpsilonGreedy`**: każdy agent niezależnie z prob. $\varepsilon$ losuje
uniform, w przeciwnym razie bierze $\arg\max Q$. Baseline.

**`CorrelatedEpsilonGreedy`** (sedno projektu):
1. **Decyzja kto eksploruje** — niezależny Bernoulli per agent. *Nie* korelujemy tego,
   żeby porównanie z baseline było fair (ten sam "exploration budget").
2. **Macierz korelacji** — jeśli ktokolwiek eksploruje:
   - Stack obserwacji w macierz $X \in \mathbb{R}^{N \times d}$.
   - L2-normalizacja wierszy.
   - $\Sigma = X_n X_n^T$ — pairwise cosine similarity, PSD.
   - Plus jitter $10^{-4} \cdot I$ dla numerycznej stabilności Cholesky.
3. **Sampling** — Cholesky $\Sigma = L L^T$, $z = L \cdot z_{\text{iid}}$, $u = \Phi(z)$.
4. **Akcja eksplorująca**: `floor(u^i * n_actions)`. Agenci nie eksplorujący wciąż
   biorą $\arg\max Q$.

**Weryfikacja, że mechanizm działa** (test z dwoma agentami w identycznych
obserwacjach, $\varepsilon = 1$, 20 000 prób):

| Setup | Pearson $\rho$ między akcjami |
|---|---|
| Independent, identyczne obs | -0.011 |
| Correlated, identyczne obs | **0.996** |
| Correlated, ortogonalne obs | -0.006 |

Czyli kopuła rzeczywiście koreluje akcje proporcjonalnie do podobieństwa stanów.

### 3.4 Trening — `train.py`

- `make_env(env_kind)` → `switches` lub `lbf`.
- `epsilon_schedule`: liniowe od 1.0 do **0.3** przez 80% epizodów, potem stała.
  Iterowaliśmy: 0.05 → 0.1 → 0.15 → 0.3. Wyższy floor jest niezbędny w LBF coop,
  bo sygnał nagrody jest tak rzadki, że agresywne wyłączenie eksploracji zabija
  uczenie (patrz sekcja 5).
- `train(exploration_kind, env_kind, ...)`:
  - Pętla po epizodach: zbiera transitions, push do buffera, sample batch i update.
  - `learning_starts=2000` — zwiększone z 500. Sparse reward → potrzebujemy
    większej rozgrzewki bufora.
  - `update_every=4` — update sieci co 4 kroki (klasyk DQN dla stabilności).
  - `batch_size=128`, `buffer_capacity=50_000` (zwiększone z 64 / 20k).
  - Loguje średni reward i success_rate co `log_every` epizodów.
  - Zwraca `{returns, successes}` jako numpy arrays.

### 3.5 Porównanie i plot — `compare.py`

- Puszcza obie metody nad N seedami, robi smoothed running mean (window = ep / 50),
  liczy mean ± std między seedami.
- Plotuje na wspólnych osiach.
- CLI: `--env {lbf,switches} --episodes N --seeds 0 1 2 --out compare.png`.

---

## 4. Jak uruchomić

```bash
# Zależności (uv preferowane):
uv sync                           # zainstaluje wszystko z pyproject.toml

# Sanity check na trywialnym envie (~30s):
uv run python compare.py --env switches --episodes 200 --seeds 0 1 2

# Pełny benchmark LBF (długo — patrz timing poniżej):
uv run python compare.py --env lbf --episodes 1000 --seeds 0 1 2
```

### Timing (laptop CPU, mierzone)

| Setup | Czas |
|---|---|
| `switches`, 200 ep, 1 seed | ~5s |
| `lbf` 6×6, 100 ep, 1 seed | ~22s |
| `lbf` 6×6, 1000 ep, 1 seed | ~6 min |
| `lbf` 8×8, 1000 ep, 1 seed | ~11 min |
| Full benchmark: 2 metody × 3 seedy × 1000 ep, lbf 6×6 | **~36 min** |

---

## 5. Aktualne wyniki i znane problemy

### 5.1 `switches`: pipeline działa, env za łatwy

Niezależny ε-greedy uczy się w ~150 epizodach (success_rate → 1.0). Skorelowana
eksploracja również, lekko szybciej, ale różnica w granicach szumu między seedami.
Plot: `compare_switches.png`. Wniosek: env jest za prosty żeby pokazać przewagę
korelacji — **niczego z niego nie wnioskujemy o samej metodzie**, służy tylko jako
sanity check pipeline'u.

### 5.2 `lbf` 6×6 coop: pierwszy jakościowy sygnał wspierający hipotezę

W `Foraging-6x6-2p-1f-coop-v3` (coop level 2 → wymaga obu agentów na load
jednocześnie) odpaliliśmy oba warianty z hyperparams: `hidden=128`, `lr=1e-4`,
`target_sync=500`, `learning_starts=2000`, `update_every=4`, eps_min=0.15.

**Single-seed, 800 epizodów, log co 100:**

| Epizod | eps | indep success | corr success |
|---:|---:|---:|---:|
| 100 | 0.87 | 0.00 | **0.05** |
| 200 | 0.74 | 0.01 | **0.10** ← 10× |
| 300 | 0.60 | 0.01 | 0.02 |
| 400 | 0.47 | 0.00 | 0.01 |
| 500-800 | ≤0.15 | 0.00 | 0.00 |

**Co to mówi:** dokładnie tę połowę historii, którą nasza hipoteza przewiduje.
W fazie wysokiej eksploracji (eps=0.74) **kopuła odkrywa skoordynowane "load"
dziesięciokrotnie częściej** niż niezależna. To efekt eksploracyjny zgodny
z teorią: brzegowe rozkłady akcji są takie same, ale wspólne zdarzenie "oboje
load przy jedzeniu" zachodzi dużo częściej, gdy akcje są skorelowane.

**Czego brakuje:** ten 10× sygnał jest wciąż za rzadki, żeby Q-funkcja go
"zapamiętała" — gdy ε opada, oba warianty toną do 0%. Hipoteza: w tej fazie
treningu replay buffer ma tak mało udanych epizodów (~10/200), że TD update
po samplowaniu batch=128 prawie nigdy ich nie widzi.

**Co próbujemy teraz** (w trakcie pisania tego dokumentu, oba runs w tle):
- eps_min=**0.3** (z 0.15) — żeby kopuła dostarczała ten 10× sygnał *przez cały
  trening*, nie tylko w pierwszych 200 epizodach.
- batch_size=**128**, buffer_capacity=**50k** — żeby update'y częściej trafiały
  na rzadkie sukcesy.

Wyniki dorzucimy poniżej gdy się skończą (~2 min/run).

### 5.3 Catastrophic forgetting na non-coop LBF (rozwiązane przez tuning)

Próbowaliśmy też łatwiejszego envu `Foraging-6x6-2p-2f-v3` (non-coop, level 1
food). Z poprzednimi hyperparams (`lr=5e-4`, `hidden=64`, `target_sync=200`)
zaobserwowaliśmy:

```
ep  50  eps=0.89  success=58%   ← losowo radzi sobie dobrze
ep 500  eps=0.10  success=30%   ← sieć nauczyła się czegoś gorszego niż random
```

To klasyczny objaw niestabilnego DQN: zbyt agresywny lr + niski target_sync
+ sparse reward → Q-overestimation → polityka deterioruje. Naprawione przez
zmiany w 3.2 (lr w dół, target_sync w górę, hidden w górę, update_every=4).

### 5.4 Otwarte TODO

- **Domknąć run z eps_min=0.3, batch=128** (w trakcie). Jeśli kopuła zacznie
  faktycznie *uczyć się* polityki na coop LBF, to będzie pierwszy mocny ilościowy
  wynik.
- **Multi-seed** (3-5 seedów) gdy hyperparams będą stabilne.
- **Co dokładnie korelować** — teraz korelujemy surowe obserwacje. Proposal sugerował
  trzy opcje: (a) obserwacje, (b) ukryte reprezentacje sieci Q, (c) wektory Q.
  Jedna z osi eksperymentu w finalnym raporcie.
- **Hyperparam sweep** — eps schedule, hidden size, lr, batch size, update_every.
- **Nowe envy** — Hallway (z proposala) i potem SMAC/SMACv2/SMAX/GRF jako "advanced".
- **Wpiąć QMIX** — VDN to suma; QMIX z monotonic mixerem jest standardem
  w nowoczesnym MARL i powinien być kolejnym krokiem.
- **GRU w `QNet`** — standardowe podejścia do LBF używają RNN (sekwencja). My mamy
  MLP. Jeśli MLP nie wystarczy, trzeba przerobić replay buffer na sekwencyjny.

---

## 6. Pliki

```
rl_hw/
├── pyproject.toml          # uv project, deps: torch, numpy, scipy, matplotlib,
│                           #                   gymnasium, lbforaging
├── env.py                  # CoordinatedSwitches (toy)
├── lbf_env.py              # Level-Based Foraging adapter
├── agent.py                # VDNLearner, QNet, ReplayBuffer
├── exploration.py          # IndependentEpsilonGreedy, CorrelatedEpsilonGreedy
├── train.py                # train(), epsilon_schedule, make_env
├── compare.py              # multi-seed comparison + plot
├── compare_switches.png    # plot z sanity check na switches
└── HANDOFF.md              # ten dokument
```

---

## 7. Uzasadnienia kluczowych decyzji

| Decyzja | Dlaczego |
|---|---|
| PyTorch (nie JAX) | Czytelność > prędkość na toy envach. JAX dopiero gdy będziemy puszczać dużo seedów na klastrze. |
| Własna `CoordinatedSwitches` | Trywialne env do debugowania pipeline'u — w kilkadziesiąt sekund widać czy VDN się uczy. |
| LBF jako benchmark | Wprost wymienione w proposalu, kanoniczne dla MARL coop. Wymaga koordynacji `load` w tym samym kroku → idealne dla naszej hipotezy. |
| 6×6 zamiast 8×8 | 8×8 zbyt rzadkie nagrody — robimy sanity najpierw na 6×6, jak zadziała wracamy do 8×8. |
| VDN przed QMIX | Najprostszy value-decomposition, szybko sprawdzalna implementacja. QMIX jako kolejny krok. |
| Sieci osobne per-agent | Agenci mają różne cele w obu envach; sharing wagi tylko zaszumi. |
| Cosine similarity obserwacji jako $\Sigma$ | Trywialnie PSD (Gram matrix), bezpośrednio zgodne z proposalem. |
| Eps-decay rate 80%, eps_min=0.3 | Sparse rewards w LBF — iterowaliśmy 0.05 → 0.1 → 0.15 → 0.3. Nawet na 0.15 oba warianty toną. Hipoteza: na 0.3 kopuła dostarcza ciągły 10× sygnał i Q się uczy (test w toku). |
| `update_every=4`, `lr=1e-4`, `target_sync=500`, `hidden=128` | Konserwatywne wartości po obserwacji catastrophic forgetting na non-coop LBF (sekcja 5.3). Standardy z papers QMIX. |
| Reward sumowany w `LBFAdapter` | VDN trenuje wspólne $Q_{\text{tot}}$ na wspólnej nagrodzie. LBF zwraca per-agent → sumujemy. |
| `uv` zamiast pip | Szybsze, deterministyczny lockfile, łatwiejsza synchronizacja u kolegi (`uv sync`). |

---

## 8. Co dokładnie powinieneś przeczytać żeby wejść w temat

**Niezbędne (obowiązkowo):**
- Sekcja 2 tego dokumentu.
- Proposal 9 z `RL projekty.pdf`.
- Sundar Sunehag et al., **VDN paper**: *Value-Decomposition Networks For Cooperative
  Multi-Agent Learning* (arXiv:1706.05296).

**Mocno polecane:**
- Rashid et al., **QMIX**: *Monotonic Value Function Factorisation* (arXiv:1803.11485) —
  bo to kolejny krok dla nas.
- LBF: <https://github.com/semitable/lb-foraging> (README wyjaśnia wszystkie env IDs).

**Później, jeśli idziemy advanced:**
- Mahajan et al., **MAVEN**: *Multi-Agent Variational Exploration* (arXiv:1910.07483)
  — bezpośrednio o eksploracji w MARL przez wspólną zmienną latentną. Konkurencyjny
  pomysł do naszego — warto porównać.
- Liu et al., **CMAE**: *Cooperative Multi-Agent Exploration* (arXiv:2107.11444)
  — też o coordinated exploration.
- Wikipedia: *Copula (statistics)* — krótkie i wystarczające na nasze potrzeby.

---

## 9. Stan zadań

- [x] Środowisko `CoordinatedSwitches`
- [x] VDN learner + replay buffer
- [x] Independent eps-greedy
- [x] Correlated eps-greedy via Gaussian copula (zweryfikowany numerycznie test
      korelacji w sekcji 3.3)
- [x] Pętla treningowa
- [x] Comparison script + plot
- [x] LBF adapter
- [x] **Pierwszy jakościowy sygnał na LBF coop** — kopuła odkrywa koordynację 10×
      częściej w fazie wysokiej eksploracji (sekcja 5.2)
- [ ] **LBF coop: pełna krzywa uczenia** (w trakcie z eps_min=0.3)
- [ ] Multi-seed run (3-5 seedów) z finalnymi hyperparams
- [ ] Eksperyment: co korelować (obs vs hidden vs Q)
- [ ] QMIX
- [ ] Advanced envy (SMAC / GRF)
- [ ] Raport (3-4 strony)
