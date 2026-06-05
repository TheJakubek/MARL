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

**QMIX** uogólnia VDN do monotonicznego mixera. Zamiast prostej sumy:
$$Q_{\text{tot}} = f_{\text{mix}}(Q_1, \dots, Q_N; s), \quad \frac{\partial f_{\text{mix}}}{\partial Q_i} \geq 0$$
gdzie $f_{\text{mix}}$ to mała sieć neuronowa z wagami pochodzącymi z **hypernetworka**
zależnego od globalnego stanu $s$. Nieujemność wag (przez `abs()`) gwarantuje, że
ta sama własność `argmax` co w VDN dalej zachodzi → decentralizacja przy wykonaniu
zachowana. QMIX jest ścisłym uogólnieniem VDN (potrafi reprezentować strictly więcej
gier kooperacyjnych — w szczególności takie, gdzie wkład agenta zależy
*kontekstowo* od stanu).

W projekcie zaimplementowaliśmy oba: VDN (`SumMixer`) i QMIX (`QMixer`). Mixer jest
wymienny przez `mixer_kind` w `MARLLearner`.

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

### 3.2 Learner — `agent.py`

- **`QNet`**: MLP `obs_dim → 128 → 128 → n_actions`, ReLU. Architektura jest
  rozdzielona:
  - `backbone`: `obs → Linear(128) → ReLU → Linear(128) → ReLU` daje
    **hidden representation** $h \in \mathbb{R}^{128}$.
  - `head`: `Linear(hidden, n_actions)` produkuje Q-wartości.
  - `forward(x, return_hidden=False)` opcjonalnie zwraca też $h$ (dla strategii
    eksploracji która koreluje po hidden).
  (Domyślne 64 okazało się za małe na LBF — zwiększyliśmy do standardowego
  rozmiaru z papers QMIX.)
- **`ReplayBuffer`**: deque o pojemności `capacity`, `push` zapisuje krotki
  `(obs, actions, reward, next_obs, done)`. `sample(batch)` zwraca tensory PyTorcha
  o kształcie `(B, n_agents, ...)`.
- **`SumMixer`**: realizuje VDN ($Q_{\text{tot}} = \sum_i Q_i$). Bez parametrów,
  bez state input.
- **`QMixer`**: monotoniczny mixer QMIX z hypernetworkiem.
  - 2-warstwowa sieć: `(B, 1, n_agents) → embed_dim → 1`.
  - Wagi pochodzą z hypernetów `hyper_w1`, `hyper_w2` zasilanych globalnym stanem;
    biasy podobnie. Nieujemność wag wymuszona przez `abs()`.
  - Domyślne `embed_dim=16`, `hyper_hidden=32` (mniejsze niż w paperze, bo nasze
    envy są małe i większy mixer zaczynał divergeować — patrz sekcja 5).
  - **State proxy** = konkatenacja per-agent obs (bo nasze envy nie wystawiają
    explicitnie globalnego stanu).
- **`MARLLearner`** (zastąpił dawny `VDNLearner`):
  - `q_nets`: `nn.ModuleList` — osobna sieć per agent (nie sharing — agenci mają
    różne cele).
  - `mixer`: `SumMixer` lub `QMixer`, wybierane przez `mixer_kind`.
  - `target_nets` + `target_mixer`: zamrożone kopie, sync co 500 update'ów.
  - `device`: auto-detect `cuda`/`cpu`. Wszystkie tensory `.to(device)` w `update()`.
  - `lr=1e-4` (zmniejszone z 5e-4 po obserwacji catastrophic forgetting na LBF).
  - `update(batch)`:
    - Per-agent: $Q_i(o^i, a^i)$ przez `gather`, stack do `(B, n_agents)`.
    - $Q_{\text{tot}}$ = `mixer(q_per_agent, state)`.
    - Target: $r + \gamma \cdot$ `target_mixer(target Q_max, next_state)`.
    - Loss = MSE.
    - Adam, gradient clipping (max_norm=10).
  - `q_values(obs_list, return_hidden=False)`: helper bez gradientu, zwraca
    numpy. Z `return_hidden=True` zwraca dodatkowo per-agent hidden activations
    $h_i \in \mathbb{R}^{128}$ (output `backbone`, przed head). Tensor leci na
    `device` i wraca na CPU dla strategii eksploracji.

Backward-compat alias `VDNLearner = MARLLearner` zachowany (mixer_kind="vdn"
domyślny).

### 3.3 Strategie eksploracji — `exploration.py`

Obie strategie mają jeden interfejs:
`select(q_values_list, obs_list, epsilon, hidden_list=None) -> list[int]`.
`hidden_list` jest opcjonalny — wymagany tylko gdy `similarity_source="hidden"`.
`IndependentEpsilonGreedy` ignoruje `obs_list` i `hidden_list` (ten sam
interfejs dla wymiennej kompatybilności w pętli treningowej).

**`IndependentEpsilonGreedy`**: każdy agent niezależnie z prob. $\varepsilon$ losuje
uniform, w przeciwnym razie bierze $\arg\max Q$. Baseline.

**`CorrelatedEpsilonGreedy`** (sedno projektu):
1. **Decyzja kto eksploruje** — niezależny Bernoulli per agent. *Nie* korelujemy tego,
   żeby porównanie z baseline było fair (ten sam "exploration budget").
2. **Wybór wektora cech** (`similarity_source`):
   - `"obs"` (domyślne) — surowe obserwacje. Koreluje agentów w *podobnych
     pozycjach na świecie*. Najprostsze, ale w LBF okazało się słabe — patrz 5.
   - `"q_values"` — wektory Q. Koreluje agentów którzy mają *podobne preferencje
     akcji* (similar intent), niezależnie od pozycji. Bezpośrednio celuje w
     intencje, ale ma chicken-and-egg: na początku Q jest losowe.
   - `"hidden"` — **ukryte reprezentacje sieci Q** ($h \in \mathbb{R}^{128}$,
     output `QNet.backbone`). Koreluje agentów w *semantycznie podobnych
     sytuacjach* zgodnie z reprezentacją wyuczoną przez sieć. Argument
     teoretyczny: $h$ jest trenowany tak, by Q dobrze przewidywał — czyli musi
     zakodować to co naprawdę wpływa na decyzje. Cosine na $h$ powinno więc
     odpowiadać "podobne sytuacje strategiczne".
3. **Macierz korelacji** — jeśli ktokolwiek eksploruje:
   - Stack wektorów cech w macierz $X \in \mathbb{R}^{N \times d}$.
   - L2-normalizacja wierszy.
   - $\Sigma = X_n X_n^T$ — pairwise cosine similarity, PSD.
   - Plus jitter $10^{-4} \cdot I$ dla numerycznej stabilności Cholesky.
4. **Sampling** — Cholesky $\Sigma = L L^T$, $z = L \cdot z_{\text{iid}}$, $u = \Phi(z)$.
5. **Akcja eksplorująca**: `floor(u^i * n_actions)`. Agenci nie eksplorujący wciąż
   biorą $\arg\max Q$.

**Weryfikacja, że mechanizm działa** (test z dwoma agentami, $\varepsilon = 1$,
20 000 prób):

Test 1 — `similarity="obs"`:
| Setup | Pearson $\rho$ między akcjami |
|---|---|
| Independent, identyczne obs | -0.011 |
| Correlated(obs), identyczne obs | **0.996** |
| Correlated(obs), ortogonalne obs | -0.006 |

Test 2 — `similarity="q_values"` (klucz: kiedy obs są ortogonalne, ale Q identyczne):
| Setup | Pearson $\rho$ między akcjami |
|---|---|
| Correlated(obs), Q identyczne, obs ortogonalne | 0.000 (degeneruje!) |
| Correlated(q_values), Q identyczne, obs ortogonalne | **0.996** |
| Correlated(q_values), Q przeciwne, obs ortogonalne | **-0.996** |

Test 3 — `similarity="hidden"` (świeża, nieuczona sieć, identyczne obs):
| Setup | hidden cosine | Pearson $\rho$ między akcjami |
|---|---:|---:|
| Correlated(hidden), identyczne obs, świeża sieć | 0.379 | **0.338** |
| Correlated(hidden), różne obs, świeża sieć | 0.198 | 0.176 |

**Uwaga o "hidden"**: na świeżej sieci hidden cosine ≈ 0.4 nawet dla identycznych
obs, **bo każdy agent ma osobną sieć** — losowe inicjalizacje dają różne hidden
nawet dla tego samego inputu. Po treningu sieci powinny zbiec do bardziej
spójnych reprezentacji (przez wspólny TD error). Z parameter sharing hidden cosine
naturalnie zbiegłby do 1.0 dla identycznych obs.

**Kluczowa obserwacja:** trzy źródła similarity korelują *różne* rzeczy. Dla LBF
ma to praktyczne znaczenie — patrz sekcja 5.

### 3.4 Trening — `train.py`

- `make_env(env_kind)` → `switches` lub `lbf`.
- `epsilon_schedule`: liniowe od 1.0 do **0.3** przez 80% epizodów, potem stała.
  Iterowaliśmy: 0.05 → 0.1 → 0.15 → 0.3. Wyższy floor jest niezbędny w LBF coop,
  bo sygnał nagrody jest tak rzadki, że agresywne wyłączenie eksploracji zabija
  uczenie (patrz sekcja 5).
- `train(exploration_kind, env_kind, mixer_kind, similarity_source, ...)`:
  - Pętla po epizodach: zbiera transitions, push do buffera, sample batch i update.
  - Wykrywa `need_hidden = (exploration_kind == "correlated" and
    similarity_source == "hidden")`. Tylko wtedy wywołuje
    `learner.q_values(obs, return_hidden=True)` — bez tego hidden nie jest
    liczone (zero overhead na innych konfiguracjach).
  - `learning_starts=2000` — zwiększone z 500. Sparse reward → potrzebujemy
    większej rozgrzewki bufora.
  - `update_every=4` — update sieci co 4 kroki (klasyk DQN dla stabilności).
  - `batch_size=128`, `buffer_capacity=50_000` (zwiększone z 64 / 20k).
  - Loguje średni reward i success_rate co `log_every` epizodów.
  - Zwraca `{returns, successes}` jako numpy arrays.

### 3.5 Single-run skrypt — `long_run.py`

CLI dla pojedynczego treningu. Używany przez `run_grid.py` jako worker, ale można
też odpalać ręcznie. Zapisuje wyniki do `.npz` z metadanymi (config, czas).

```bash
uv run python long_run.py \
    --exploration correlated --mixer qmix --similarity q_values \
    --env lbf --episodes 5000 --seed 0 \
    --out results/lbf_qmix_correlated_q_values_s0.npz
```

Wszystkie wybory są CLI args. Output jest "live" — zapisuje co `--save-every`
epizodów (domyślnie 100), więc można obserwować postęp bez czekania na koniec.

### 3.6 Orkiestrator siatki — `run_grid.py`

Definiujesz przestrzeń (env × mixer × exploration × similarity × seeds), skrypt
odpala wszystkie kombinacje przez `ProcessPoolExecutor` z konfigurowalnym limitem
równoległości. **Idempotentne**: pomija joby które już mają kompletny `.npz`.

```bash
uv run python run_grid.py \
    --envs switches lbf \
    --mixers vdn qmix \
    --explorations independent correlated \
    --similarities obs q_values \
    --seeds 0 1 2 3 4 \
    --episodes 5000 \
    --concurrency 4
```

Output: `results/<tag>.npz` (jeden plik per job), `logs/<tag>.log` (stdout
tego joba). Tag = `<env>_<mixer>_<exploration>_<sim_or_na>_s<seed>`.

`--dry-run` pokaże planowaną listę bez odpalania.

### 3.7 Porównanie i plot — `compare.py` (legacy, do refaktoryzacji)

Aktualnie obsługuje tylko `independent` vs `correlated` (jednorazowo, bez
wyboru mixera/similarity). **Do przepisania** żeby czytał z `results/*.npz`
i robił plot per (env, mixer) z wieloma krzywymi (per exploration × similarity).
TODO post-klastrowy.

---

## 4. Jak uruchomić

### Setup

```bash
uv sync                           # instaluje wszystko z pyproject.toml
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Pojedynczy run (ręczny test, ~minuty)

```bash
uv run python long_run.py \
    --exploration correlated --mixer vdn --similarity obs \
    --env switches --episodes 500 --seed 0 \
    --out results/test.npz
```

### Cała siatka eksperymentów (klaster)

```bash
uv run python run_grid.py \
    --envs switches lbf \
    --mixers vdn qmix \
    --explorations independent correlated \
    --similarities obs q_values hidden \
    --seeds 0 1 2 3 4 \
    --episodes 5000 \
    --concurrency 4
```

Liczba jobów: `2 envs × 2 mixers × (1 indep + 3 corr×similarities) × 5 seedy
= 80 jobów`. Z `--concurrency 4` i ~5 min/job na GPU: ~100 min wall-clock.

### Timing (laptop CPU, mierzone)

| Setup | Czas |
|---|---|
| `switches`, 200 ep, 1 seed, VDN | ~5s |
| `switches`, 300 ep, 1 seed, QMIX | ~20s |
| `lbf` 5×5 coop, 800 ep, 1 seed, VDN | ~85s |
| `lbf` 6×6 coop, 800 ep, 1 seed, QMIX | ~143s |

Z GPU spodziewane: ~3-5× szybciej dla update'ów; rollouts (CPU env step) bez
zmian. Bottleneck na małych envach to env step, więc realne przyspieszenie
GPU najpewniej rzędu 2×.

---

## 5. Aktualne wyniki i znane problemy

Wyniki uporządkowane chronologicznie wg envu. Wszystkie liczby z **single-seed**
runów (laptop CPU). Multi-seed na klastrze — TODO.

### 5.1 `switches`: pipeline działa, env za łatwy do różnicowania metod

Niezależny ε-greedy uczy się w ~150 epizodach (success_rate → 1.0). Skorelowana
eksploracja również, lekko szybciej, ale różnica w granicach szumu między seedami.
QMIX uczy się porównywalnie (po skorygowaniu rozmiaru mixera — patrz 5.4).

**Wniosek:** env jest za prosty żeby pokazać przewagę korelacji. Służy jako
sanity check pipeline'u — nie wnioskujemy z niego o metodzie.

### 5.2 `lbf` 6×6 coop: jakościowy sygnał wspierający hipotezę, ale uczenie nie startuje

W `Foraging-6x6-2p-1f-coop-v3` (level 2 food → wymaga obu agentów na load
jednocześnie). Single-seed, 800 epizodów, hyperparams docelowe (hidden=128,
lr=1e-4, target_sync=500, eps_min=0.15, learning_starts=2000, update_every=4).

| Epizod | eps | VDN+indep | VDN+corr(obs) |
|---:|---:|---:|---:|
| 100 | 0.87 | 0% | **5%** |
| 200 | 0.74 | 1% | **10%** ← 10× |
| 300 | 0.60 | 1% | 2% |
| 400-800 | ≤0.47 | 0% | 0-1% |

**Co to mówi:** w fazie wysokiej eksploracji (eps=0.74) **kopuła odkrywa
skoordynowane "load" 10× częściej** niż niezależna eksploracja. Brzegowe
rozkłady akcji są takie same — różnica jest tylko w *współwystępowaniu*
"oboje load w tym samym kroku".

**Czego brakuje:** ten 10× sygnał wciąż za rzadki, żeby Q-funkcja go zapamiętała.
Gdy ε opada, oba warianty toną do 0%. Replay buffer ma ~10-20 udanych epizodów
na 200 — w batchu 128 udane transicje pojawiają się sporadycznie.

### 5.3 Próby naprawienia LBF (wszystkie z jednym seedem)

#### 5.3.1 Wyższe eps_min (0.3) + większy batch (128) + większy buffer (50k)

Hipoteza: kopuła musi dostarczać 10× sygnał *przez cały trening*, nie tylko 200 ep.

| Epizod | eps | VDN+indep | VDN+corr(obs) |
|---:|---:|---:|---:|
| 100 | 0.89 | 2% | **6%** |
| 200 | 0.78 | 3% | 3% |
| 400 | 0.56 | 0% | 2% |
| 800 | 0.30 | 0% | 0% |

**Wniosek:** efekt mniejszy niż przy eps_min=0.15 (3× zamiast 10×) — pewnie
dlatego że zwolniliśmy lr i learner ma jeszcze losowe Q. Ciągle nie uczy.

#### 5.3.2 QMIX zamiast VDN, eps_min=0.3, 800 ep

Hipoteza: bogatszy mixer lepiej rozdzieli credit assignment przy rzadkich sukcesach.

| Epizod | eps | QMIX+corr(obs) |
|---:|---:|---:|
| 100 | 0.89 | 5% |
| 200 | 0.78 | 3% |
| 400 | 0.56 | 1% |
| 800 | 0.30 | 1% |

**Wniosek:** liczby identyczne z VDN+corr. Mixer **nie** był bottleneckiem.

#### 5.3.3 Mniejszy LBF (5×5 coop) zamiast 6×6

| Epizod | eps | VDN+corr(obs) |
|---:|---:|---:|
| 100 | 0.86 | 11% |
| 300 | 0.56 | 3% |
| 600+ | ≤0.34 | 0% |

**Wniosek:** start lepszy (więcej "przypadkowych" trafień w mniejszej siatce),
ale ten sam wzorzec — sygnał ginie gdy ε opada.

#### 5.3.4 Q-values zamiast obs jako similarity source

Hipoteza: w LBF agenci są blisko siebie tylko *w momencie sukcesu*. Gdy są
daleko, cosine(obs_i, obs_j) jest niska → korelacja akcji niska → kopuła
degeneruje do niezależnego eps-greedy. Korelacja po Q-wartościach koreluje
*intencje* niezależnie od pozycji.

5x5 coop, 800 ep, VDN, eps_min=0.3:

| Epizod | eps | corr(obs) | corr(q_values) |
|---:|---:|---:|---:|
| 100 | 0.89 | 6% | 6% |
| 200 | 0.78 | 5% | **7%** |
| 400 | 0.56 | 1% | 2% |
| 800 | 0.30 | 1% | 1% |

**Wniosek:** Q-values lekko lepsze, ale w tej samej skali. Wciąż nie uczy.

#### 5.3.5 Hidden representations jako similarity source

Trzecia (i ostatnia z proposala) opcja. Argument: surowe obs nie kodują semantyki
(np. `(3,2)` i `(3,4)` to obaj sąsiedzi jedzenia w `(3,3)` — ale liczbowo różni),
Q-values są zbyt aliasowy (6 wymiarów dla LBF). Hidden layer Q-funkcji jest
trenowany tak, by wyciągać semantycznie ważne cechy stanu — cosine na hidden
powinien lepiej kodować "podobne sytuacje strategiczne".

`QNet` zostało rozdzielone na `backbone` (obs → 128) + `head` (128 → n_actions).
Hidden = output `backbone`, $h \in \mathbb{R}^{128}$.

**Sanity check (świeża sieć, jeszcze nieuczona)**, switches 200 ep, single seed:

| Similarity source | Success rate (last 50 ep) |
|---|---:|
| obs | 0.88 |
| q_values | 0.62 |
| hidden | **0.94** |

Mechanizm działa, ale single-seed na switches niewiele mówi (env za łatwy
+ losowość seedów dominuje). **LBF z hidden similarity** — TODO, multi-seed
na klastrze.

**Niuans:** na **świeżej** sieci hidden cosine ≈ 0.4 nawet dla identycznych obs,
bo każdy agent ma osobną sieć z innymi losowymi wagami. Po treningu sieci powinny
zbiec do bardziej spójnych reprezentacji. To jest argument za parameter sharing
w przyszłej pracy — z dzielonymi wagami hidden cosine startowałby od 1.0 dla
identycznych obs, dając kopule "darmowy boost" na samym starcie.

### 5.4 QMIX: drobny gotcha z rozmiarem mixera

Pierwsza implementacja używała `embed_dim=32`, `hyper_hidden=64` (wartości z
papera dla SMAC z 8+ agentami). Na switches QMIX nie uczył się 600 ep.
Diagnostyka pokazała:
- Loss spada poprawnie.
- Q_tot się zmienia.
- Monotoniczność spełniona.

Przyczyna: mixer miał ~6000 parametrów dla 2 agentów i obs_dim=6 — duży overfit
na małej ilości danych. Po zmniejszeniu do `embed_dim=16`, `hyper_hidden=32`
QMIX uczy switches na poziomie VDN. To wartość domyślna w `agent.py:QMixer`.

### 5.5 Catastrophic forgetting na non-coop LBF (rozwiązane przez tuning)

Próbowaliśmy `Foraging-6x6-2p-2f-v3` (non-coop, level 1 food) z poprzednimi
hyperparams (`lr=5e-4`, `hidden=64`, `target_sync=200`):

```
ep  50  eps=0.89  success=58%   ← losowo radzi sobie dobrze
ep 500  eps=0.10  success=30%   ← sieć nauczyła się czegoś gorszego niż random
```

Klasyczny niestabilny DQN: agresywny lr + niski target_sync + sparse reward →
Q-overestimation → feedback loop. Naprawione przez `lr=1e-4`, `target_sync=500`,
`hidden=128`, `update_every=4`. Te wartości stały się domyślne.

### 5.6 Synteza wniosków po dzisiaj

1. **Mechanizm kopuły działa zgodnie z teorią** — zweryfikowane numerycznie.
   Korelacja akcji w eksploracji odpowiada cosine similarity wektorów cech.

2. **Na łatwym envie (switches) różnica jest niewidoczna** — bo niezależny
   ε-greedy też rozwiązuje. Przewaga kopuły mierzalna byłaby tylko w sample
   efficiency (wymaga multi-seed).

3. **Na trudnym envie (LBF coop) widzimy 3-10× przewagę eksploracyjną**
   (więcej skoordynowanych zdarzeń), ale **żaden bazowy learner nie potrafi tego
   wykorzystać** w 800 epizodach — niezależnie od mixera (VDN/QMIX), wielkości
   grida (5×5/6×6), poziomu eps_min ani similarity source (obs/q_values/hidden).
   Pełen multi-seed ablation similarity to zadanie post-klastrowe.

4. **Bottleneck nie leży w eksploracji ani mixerze** — leży gdzieś indziej.
   Najpewniejsi kandydaci: brak GRU (LBF wymaga pamięci), brak parameter
   sharing (2× efektywnych danych byłoby przy współdzieleniu sieci), za mało
   epizodów (papers MARL trenują 200k+ epizodów, my trenowaliśmy 800).

### 5.7 Otwarte TODO

**Wymagane do raportu:**
- Multi-seed run pełnej siatki (env × mixer × exploration × similarity × seeds)
  na klastrze — `run_grid.py` jest gotowy.
- Plot: per (env, mixer), krzywe per (exploration, similarity), mean ± std
  z N seedów. Skrypt do napisania.

**Pomysły na rozszerzenie hipotezy:**
- **Parameter sharing** w `QNet` — standard w MARL, 2× więcej danych per agent.
  ~20 linii kodu. Mogłoby też pomóc hidden similarity (bo z osobnymi sieciami
  hidden cosine na świeżej sieci nie startuje od 1.0 nawet dla identycznych obs).
- **GRU** w `QNet` — pamięć między krokami. Wymaga przerobienia replay buffera
  na sekwencyjny. ~1-2h pracy.
- **Dynamiczny mix similarity** — np. `obs` na początku treningu (gdy hidden/Q
  jest losowe), `hidden` po jakimś warm-upie. Adresuje chicken-and-egg z 5.3.4.

**Pomysły na ekspansję ewaluacji:**
- **Hallway** — drugi toy env z proposala, zaprojektowany dokładnie do
  testowania koordynacji. Może być łatwiejszy niż LBF.
- **SMAC/SMACv2/GRF** — jako "advanced stage" z proposala. Tylko jeśli
  poprzednie kroki działają.

---

## 6. Pliki

```
rl_hw/
├── pyproject.toml          # uv project, deps: torch, numpy, scipy, matplotlib,
│                           #                   gymnasium, lbforaging
├── env.py                  # CoordinatedSwitches (toy)
├── lbf_env.py              # Level-Based Foraging adapter (default 5x5-2p-1f-coop)
├── agent.py                # MARLLearner (VDN/QMIX), QNet, SumMixer, QMixer,
│                           # ReplayBuffer; GPU-aware
├── exploration.py          # IndependentEpsilonGreedy, CorrelatedEpsilonGreedy
│                           # (similarity_source: obs | q_values)
├── train.py                # train(), epsilon_schedule, make_env
├── long_run.py             # CLI single-run, .npz output, used by run_grid
├── run_grid.py             # parallel grid orchestrator, idempotent, .npz/.log per job
├── compare.py              # legacy multi-seed comparison + plot (do refaktoryzacji)
├── results/                # .npz output per job (created by run_grid)
├── logs/                   # stdout per job (created by run_grid)
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
| `update_every=4`, `lr=1e-4`, `target_sync=500`, `hidden=128` | Konserwatywne wartości po obserwacji catastrophic forgetting na non-coop LBF (sekcja 5.5). Standardy z papers QMIX. |
| QMIX: `embed_dim=16`, `hyper_hidden=32` (mniej niż w paperze) | Paper miał 32/64 dla SMAC z 8+ agentami i obs_dim ~30. Dla naszych 2 agentów i obs_dim 6-9, większy mixer overfittuje. Patrz 5.4. |
| Mixer pluggable (`SumMixer` / `QMixer`) | Pozwala ablate "wkład mixera" oddzielnie od "wkładu eksploracji" — kluczowe dla raportu, w którym chcemy odpowiedzieć "co jest bottleneckiem?". |
| `similarity_source` configurable (`obs` / `q_values` / `hidden`) | Wszystkie trzy źródła similarity wymienione w proposalu są zaimplementowane. Pozwala na ablację — porównujemy *gdzie* leży problem (kodowanie pozycji vs intencje vs uczone reprezentacje). |
| Hidden similarity — `QNet` rozdzielony na `backbone`+`head` | Pozwala wystawiać $h$ z penultimate layer bez podwójnego forward pass. Hidden_dim=128 (rozmiar warstwy ukrytej). |
| State proxy = concat(obs) | Nasze envy nie wystawiają explicit globalnego stanu. Konkat obs to standard w wielu implementacjach SMAC bez globalnego stanu. |
| GPU support przez auto-detect device | Lokalnie CPU, na klastrze CUDA — bez zmiany kodu. |
| `run_grid.py` zamiast bash for-loop | Idempotentne (skipuje ukończone), kontrola równoległości, automatyczny logging per-job. Nadaje się 1:1 na klaster. |
| `.npz` jako format wyników | Zawiera krzywe + config + metadata w jednym pliku, łatwo wczytać do plotów. |
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
  zaimplementowane w `agent.py:QMixer`.
- Wang et al., **QPLEX**: *Duplex Dueling Multi-Agent Q-Learning* (arXiv:2008.01062)
  — dueling + attention, IGM-complete. Ewentualne rozszerzenie.
- Liu et al., **NA²Q**: *Neural Attention Additive Q-Learning* (arXiv:2304.13383)
  — additive nonlinear z attention, interpretowalny.
- LBF: <https://github.com/semitable/lb-foraging> (README wyjaśnia wszystkie env IDs).

**O eksploracji w MARL (konkurencyjne pomysły):**
- Mahajan et al., **MAVEN**: *Multi-Agent Variational Exploration* (arXiv:1910.07483)
  — eksploracja przez wspólną zmienną latentną. Główny benchmark to Hallway,
  który jest też w naszym proposalu.
- Liu et al., **CMAE**: *Cooperative Multi-Agent Exploration* (arXiv:2107.11444)
  — coordinated exploration przez wspólne curriculum.
- Wikipedia: *Copula (statistics)* — krótkie i wystarczające na nasze potrzeby.

**Przydatne benchmarki MARL z hyperparams:**
- Papoudakis et al., **EPyMARL**: *Benchmarking Multi-Agent Deep RL Algorithms*
  (arXiv:2006.07869) — kanoniczne porównanie algorytmów na LBF/SMAC, dobre
  wartości hyperparams z RNN/parameter sharing.

---

## 9. Stan zadań

**Done:**
- [x] Środowisko `CoordinatedSwitches`
- [x] LBF adapter (default 5×5-2p-1f-coop)
- [x] VDN learner + replay buffer
- [x] **QMIX** mixer + hypernetwork (monotoniczny, parametryzacja przez state)
- [x] Wymienny mixer (`SumMixer` / `QMixer`) w `MARLLearner`
- [x] Independent eps-greedy
- [x] Correlated eps-greedy via Gaussian copula (zweryfikowany numerycznie)
- [x] **Q-values jako similarity source** (drugi z trzech wymienionych w proposalu)
- [x] **Hidden representations jako similarity source** (trzeci z trzech;
      `QNet` rozdzielone na backbone+head, sanity check OK na switches)
- [x] Pętla treningowa
- [x] **GPU support** (auto-detect device, tensory `.to(device)` w update/q_values)
- [x] **`long_run.py`** — single-run CLI z .npz output
- [x] **`run_grid.py`** — orkiestrator równoległych eksperymentów, idempotentny
- [x] **Pierwszy jakościowy sygnał na LBF** — kopuła odkrywa koordynację 3-10×
      częściej (sekcja 5.2). Mechanizm działa zgodnie z teorią.

**Wymagane do raportu:**
- [ ] Multi-seed pełnej siatki na klastrze (czeka na dostęp do Entropy)
- [ ] Plot script: `results/*.npz` → wykresy per (env, mixer)
- [ ] Raport (3-4 strony) — dwa wątki:
      (a) mechanizm kopuły jest poprawny, (b) bottleneck w LBF nie leży w
      eksploracji ani mixerze

**Pomysły rozszerzające (jeśli zostaje czas):**
- [ ] Hidden representations jako trzeci similarity source
- [ ] Parameter sharing w `QNet`
- [ ] GRU w `QNet` (replay buffer na sekwencje)
- [ ] Hallway env (drugi toy z proposala)
- [ ] SMAC / GRF (advanced stage z proposala)
- [ ] QPLEX / NA²Q (advanced mixers z proposala)
