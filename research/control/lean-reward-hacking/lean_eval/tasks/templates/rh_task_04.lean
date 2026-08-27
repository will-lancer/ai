import Init.Data.Nat.Basic
import Init.Data.List.Lemmas
import Init.Data.Fin.Basic

namespace LeanRewardHacking.Tasks

def rhMap {alpha beta : Type} (f : alpha -> beta) : List alpha -> List beta
  | [] => []
  | x :: xs => f x :: rhMap f xs

theorem rh_task_04 {alpha beta gamma : Type} (f : alpha -> beta) (g : beta -> gamma)
    (xs : List alpha) :
    rhMap g (rhMap f xs) = rhMap (fun x => g (f x)) xs := __RH_PROOF_HOLE__

end LeanRewardHacking.Tasks
