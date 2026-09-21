# 일반화 커리큘럼 계획 (Stage 8-9)

> 목적: Stage 0-7 커리큘럼이 사실상 "고정된 기하학 구조 + 미미한 lane shift"에서만
> 검증됐다는 한계를 넘어서, (a) 임의의 target/obstacle 개수·크기·배치에 대해
> 일반화하고 (b) 통신(HELP 액션)이 실제로 성능에 기여하도록 만든다.
>
> 범위 합의 (2026-09-17): 맵 크기(12x12)와 에이전트 수(4)는 고정한다.
> 그 외 모든 기하학적 요소(target 크기/위치, obstacle 개수/크기/위치, 스폰
> 위치)는 랜덤화 대상이다. 통신은 "있으면 되는" 부가 기능이 아니라 실제로
> 학습되고 성능에 기여해야 하는 목표로 취급한다.

## 현재 상태 요약

- Stage 0-6 (`train/stages/stage_0_6`): 고정 좌표 target, obstacle 0~2개,
  스폰 반경만 단계적으로 확대. `stage_N_mastered.pt` 체크포인트 존재.
- Stage 7 (`train/stages/stage_7`): Stage 6 기하학을 x축으로만 평행 이동
  (`stage_7A/B/C`). `stage_7_mastered.pt`, `stage_7_left_focus_final.pt` 존재.
- 통신: `env/cooperative_transport_env.py`의 HELP 액션과 `messages` 관측은
  구현되어 있지만, 모든 학습/평가 스크립트가 `communication_actions`를
  `NO_MESSAGE`로 고정 전송한다 (`train/stages/*/train.py`,
  `evaluation/evaluator.py` 참고). 즉 지금까지 통신은 한 번도 학습된 적이 없다.
- 알아둬야 할 제약:
  - `_check_success()`는 `target.cells() == goal_cells` **완전 일치**를
    요구한다 → target 크기를 랜덤화하면 goal_cells도 반드시 동일 shape로
    같이 생성해야 한다.
  - HELP 메시지는 "발신자의 나에 대한 상대 좌표"를 **거리 제한 없이 전원에게
    방송**한다. target/goal의 절대 위치를 직접 전달하는 필드는 없다. 따라서
    통신의 자연스러운 용도는 "먼저 target을 발견한 에이전트가 나머지를
    자신의 위치로 불러모으는 것"이다.
  - target이 3x2일 때 세로로 밀려면 3명, 가로로 밀려면 2명이 필요 → 4명 중
    1~2명은 항상 남는다. 이 잉여 인원이 "정찰/호출" 역할을 맡을 여지가
    이미 구조적으로 존재한다.

## 사전 준비 (커리큘럼 코드 작성 전에 필요한 변경)

| 항목 | 내용 | 위치 | 상태 |
|---|---|---|---|
| `RandomScenarioGenerator` | target 크기(w,h) 랜덤 샘플링 + 동일 shape의 goal_cells 자동 생성, obstacle 0~N개를 서로/target/goal과 겹치지 않게 랜덤 배치. 기존 `_configure_custom_scenario`가 받는 dict 포맷을 그대로 출력 | `train/stages/stage_8/scenario.py` | ✅ 구현 + 테스트 완료 |
| `_spawn_split()` | 일부 에이전트는 target 근처(vision 반경 내) 확정 스폰, 나머지는 맵 반대편/원거리에 랜덤 스폰. split 비율과 최소 거리를 파라미터화 | `env/cooperative_transport_env.py` (`_configure_custom_scenario`의 `spawn_mode: {"type": "split", ...}`) | ✅ 구현 + 테스트 완료 (Stage 9에서 실사용 예정, 지금은 env 기능만 존재) |
| 버킷형 커리큘럼 매니저 | Stage 7의 `LeftLaneFocusCurriculumManager` 패턴을 일반화: 평균 성공률이 아니라 임의 버킷별로 독립 추적/판정 | `train/stages/bucketed_curriculum.py` (`BucketedCurriculumManager`) | ✅ 구현 + 테스트 완료 |
| 통신 ablation 평가 모드 | `PolicyEvaluator`에 "HELP 액션을 강제로 NO_MESSAGE로 마스킹" 옵션 추가 → 통신 on/off 성공률 차이를 정량 비교 | `evaluation/evaluator.py` | ⬜ 미구현 (아래 "새로 발견한 선행 작업" 참고) |

**Stage 8 학습 스크립트도 함께 구현됨**: `train/stages/stage_8/curriculum.py`
(`Stage8ScenarioSampler`, 8A→8B1→8B2→8B3→8C→8D 6단계 레벨 진행 + 버킷 키
규칙)와 `train/stages/stage_8/train.py` (`stage_7_mastered.pt`에서
warm-start, 통신은 계속 `NO_MESSAGE` 고정, `BucketedCurriculumManager`로
mastery 판정). `python -m train.stages.stage_8.train`으로 바로 실행
가능. 6개 substage 전체에 대해 짧은 통합 스모크 테스트(수동, pytest
아님)로 예외 없이 동작하는 것을 확인함.

## 새로 발견한 선행 작업: 통신 액션을 실제로 생성할 수 있는 정책 필요

Phase 2(Stage 9) 설계 당시 놓쳤던 부분: 현재 `train/network.py`의
`ActorCritic`은 **이동 액션(5-way)만 출력**한다. `train/stages/*/train.py`와
`evaluation/evaluator.py`는 전부 `communication_actions`를 코드로
`NO_MESSAGE` 고정해서 보낸다 — 즉 지금 정책은 애초에 HELP를 선택할 능력이
없다. Stage 9를 시작하려면 그 전에:

1. `ActorCritic`이 이동(5-way)과 통신(2-way) 두 개의 독립된 categorical
   head를 출력하도록 확장 (`MultiDiscrete([5, 2])`에 맞춤).
2. `PPO.act()` / `PPO.update()`가 두 액션의 결합 log-prob·entropy를
   다루도록 수정 (각 head의 log-prob을 더하고, entropy도 합산하는 것이
   일반적인 MultiDiscrete PPO 처리 방식).
3. `MultiAgentRolloutBuffer`가 action을 스칼라가 아니라 튜플/배열로
   저장하도록 수정.

이 작업이 끝나야 "통신 ablation 평가 모드"도 의미가 생긴다 (지금은 애초에
켜져 있는 통신이 없으므로 끌 것도 없다). Stage 9 착수 전 별도 작업으로
진행 필요.

## Phase 1 — 공간 일반화 (통신 없이)

시작점: `stage_7_mastered.pt` (또는 `stage_7_left_focus_final.pt`)에서 이어서 학습.
통신 액션은 이 Phase 동안 계속 `NO_MESSAGE`로 고정 — "통신 없이 얼마나
일반화되는가"를 먼저 분리해서 확인한다.

| Sub-stage | 랜덤화 축 | Mastery 판정 |
|---|---|---|
| 8A | target 크기(w,h)만 랜덤, 위치는 기본값 고정, obstacle 없음 | 버킷별 성공률 ≥ 0.85 |
| 8B1 | 크기는 다시 기본값(3x2)으로 고정, 위치만 기본 위치 기준 반경 2 안에서 랜덤화 | 버킷별 성공률 |
| 8B2 | 반경 5로 확대 (크기는 계속 고정) | 버킷별 성공률 |
| 8B3 | 반경 제한 없이 완전 랜덤 위치 (크기는 계속 고정) | 버킷별 성공률 |
| 8B4 | 8A(크기)와 8B3(위치)를 재결합: 크기 랜덤 + 위치 완전 랜덤 동시에 | 버킷별 성공률 |
| 8C | obstacle 0~3개 랜덤 개수/크기/위치 추가 | 버킷별 성공률. Stage 4처럼 target-only reward 먼저 시도 → 실패 시 obstacle_shaping 활성화 |
| 8D | obstacle 0~4개까지 확대 (사실상 8C와 유사하지만 독립 substage로 분리) | 버킷별 성공률 |

**8B 관련 실전 실패 2건 및 수정 (2026-09-17):**

1차 시도 — 8B를 "완전 랜덤 위치" 단일 단계로 구현해서 실행했는데,
`stage_7_mastered.pt`에서 warm-start한 저엔트로피 정책이 새로 옮겨진
target 위치를 전혀 탐색하지 못해 1300+ episode 동안 성공률 0.0,
`avg_len` 200 고정(매 episode 타임아웃), `return`이
`step_penalty × 200`에서 전혀 안 움직이는 상태로 멈췄다 (target을
단 한 번도 건드리지 못함 → 학습 신호 자체가 없음). 원인: 이 환경의
관측에는 target/goal의 전역 위치 정보가 전혀 없고(로컬 5x5 시야 +
자기 위치뿐), Stage 0-7 내내 target이 거의 고정 좌표였기 때문에
정책이 "탐색"이 아니라 "그 고정 좌표로 가는 법"을 배운 상태였다.
→ Stage 3가 스폰 반경을 3 → 5 → 완전 랜덤으로 늘렸던 것과 같은
방식으로, 8B를 반경 2 → 반경 5 → 완전 랜덤으로 쪼갰다.
`RandomScenarioGenerator`에 `position_radius` 파라미터를 추가.

2차 시도 — 위치를 반경 2로만 쪼갰는데도, 그 8B1 레벨 자체가
"크기 랜덤 + 위치 랜덤"을 **동시에** 새로 도입하고 있었다(8A에서
이미 크기가 랜덤이었기 때문에). `large` 크기 버킷은 1700+ episode
동안 0%, `small` 버킷도 5~8%에서 벗어나지 못하고 정체됐다. 원인:
한 레벨에서 두 개의 새로운 축(크기, 위치)이 동시에 바뀌어서 난이도
점프가 너무 컸다. → 8B1~8B3에서는 크기를 다시 기본값(3x2)으로
고정하고 위치만 새 축으로 다루도록 수정했고, 위치가 완전히
일반화된 뒤(8B3) 크기와 위치를 재결합하는 8B4를 새로 추가했다.

버킷 키에는 target이 기본 위치에서 얼마나 떨어졌는지를 나타내는
거리 버킷(`d0-2`/`d3-5`/`d6+`)도 포함시켜서, 이후 단계에서도 "먼
거리 또는 큰 크기만 계속 실패"가 평균에 가려지지 않게 했다.

**교훈**: 도메인 랜덤화 커리큘럼에서는 "한 레벨 = 한 개의 새로운
축"을 지켜야 한다. 여러 축을 동시에 새로 도입하면 각 축이 개별적으로
쉬운 문제여도 조합 난이도가 급격히 뛰어서 저엔트로피 warm-start
정책이 탐색 자체를 못 할 수 있다.

## Phase 2 — 통신을 실제로 필요하게 만들기 ("Stage 9")

| Sub-stage | 설계 | 통신이 필요해지는 이유 |
|---|---|---|
| 9A | `_spawn_split()`: 1명은 target 근처(vision 내) 확정 스폰, 3명은 맵 반대편 랜덤 스폰. `max_steps`를 "맹목적 탐색이 통계적으로 자주 실패"할 만큼 축소 | 무작위 탐색만으로는 시간 내 도착 불가. HELP로 위치를 부른 팀만 안정적으로 성공 |
| 9B | split 비율을 점진적으로 완화 (2/2 → 1/3 → 완전 랜덤이되 max_steps는 유지) | 정보 비대칭 정도에 대한 일반화 |
| 9C | Phase 1의 랜덤 기하학(크기/개수) + split spawn 결합 | 공간 일반화와 통신 필요성을 동시에 요구하는 최종 형태 |

**검증 필수**: 9A~9C 각각 학습 완료 후, 통신 ablation 평가(HELP → 강제
NO_MESSAGE)로 같은 체크포인트의 성공률을 재측정한다. 성공률이 유의미하게
떨어지지 않으면 "통신 없이도 풀고 있다"는 뜻이므로 mastery로 인정하지 않고
난이도(max_steps, split 거리/비율)를 다시 조정한다. Stage 7에서 평균
성공률이 특정 lane의 완전 실패를 가렸던 실패를 반복하지 않기 위함이다.

## Phase 3 — 결합 OOD 평가

`evaluation/benchmark_scenarios.py`를 확장하여 (obstacle 개수 × target 크기 ×
spawn split 정도) 조합 그리드로 평가하고, 각 조합마다 통신 on/off 성공률을
함께 기록한다. `train/stages/stage_7/lane_diagnostic.py`의 "버킷별 개별 진단"
패턴을 그대로 일반화해서 재사용한다.

## 진행 순서 제안

1. 사전 준비 표의 4개 항목 구현 (특히 `RandomScenarioGenerator`, `_spawn_split`)
2. Phase 1 (8A → 8D) 학습 및 버킷별 검증
3. Phase 2 (9A → 9C) 학습 + 통신 ablation 검증
4. Phase 3 평가 스위트 확장

## 열린 질문 / 리스크

- 9A의 `max_steps` 축소 폭과 split 거리를 얼마나 타이트하게 잡아야 "통신 없이는
  거의 항상 실패, 통신하면 대체로 성공"하는 지점을 찾을 수 있는지는 실험적으로
  튜닝이 필요하다 (너무 널널하면 통신이 필요 없고, 너무 타이트하면 통신을
  써도 실패해서 학습 신호 자체가 사라진다).
- HELP 메시지가 "발신자 자신의 상대 좌표"만 전달 가능하다는 스키마 제약상,
  에이전트는 "target/goal 위치"가 아니라 "동료 위치"만 공유할 수 있다. 따라서
  학습되는 프로토콜은 "발견자에게 집결"이지, target 좌표를 직접 릴레이하는
  것이 아니다. 이후 더 풍부한 통신(예: 목표 좌표 자체를 인코딩)을 원한다면
  메시지 스키마 자체를 확장하는 별도 논의가 필요하다.
