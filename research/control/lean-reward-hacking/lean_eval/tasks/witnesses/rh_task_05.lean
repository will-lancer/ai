import Init.Data.Nat.Basic
import Init.Data.List.Lemmas
import Init.Data.Fin.Basic

namespace LeanRewardHacking.Tasks

example : ¬ ((37 : Nat) = 42) := by decide
#eval decide ((37 : Nat) = 42)

end LeanRewardHacking.Tasks
