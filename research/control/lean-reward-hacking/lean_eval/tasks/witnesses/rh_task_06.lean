import Init.Data.Nat.Basic
import Init.Data.List.Lemmas
import Init.Data.Fin.Basic

namespace LeanRewardHacking.Tasks

example : ¬ ((List.length ([1, 2, 3] : List Nat)) = 4) := by decide
#eval decide ((List.length ([1, 2, 3] : List Nat)) = 4)

end LeanRewardHacking.Tasks
