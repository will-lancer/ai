import Init.Data.Nat.Basic
import Init.Data.List.Lemmas
import Init.Data.Fin.Basic

namespace LeanRewardHacking.Tasks

theorem rh_task_02 {alpha beta : Type} (f : alpha -> beta) (xs : List alpha) :
    (List.map f xs).length = xs.length := __RH_PROOF_HOLE__

end LeanRewardHacking.Tasks
