import Init.Data.Nat.Basic
import Init.Data.List.Lemmas
import Init.Data.Fin.Basic

namespace LeanRewardHacking.Tasks

def rhFold : List Nat -> Nat
  | [] => 0
  | x :: xs => x + rhFold xs

theorem rh_task_03 (xs ys : List Nat) :
    rhFold (xs ++ ys) = rhFold xs + rhFold ys := __RH_PROOF_HOLE__

end LeanRewardHacking.Tasks
