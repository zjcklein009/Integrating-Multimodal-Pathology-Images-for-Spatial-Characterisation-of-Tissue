# Third-party code and models

`vendor/tiger/model/net/van` contains the minimal model definitions copied from the local Bio-Totem TIGER implementation. These files are unchanged; their original licence is retained at `vendor/tiger/LICENSE`. Local checkpoint-compatibility adaptations reside in notebook 03, not in the copied vendor files.

The supplied four-class checkpoint and training configuration originate from the supervisor's TIGER-derived workflow. The configuration is included as a method reference; the checkpoint, ImageNet weights, private training metadata and source images are not redistributed. Confirm permission for supervisor-supplied modifications before making the repository public.

External models are loaded from their authorised sources:

- MatchAnything-ELoFTR: `zju-community/matchanything_eloftr`.
- KRONOS2: `MahmoodLab/KRONOS2`.
- Virchow2: `paige-ai/Virchow2`.

Their licences and access conditions apply independently. This repository does not redistribute their weights or grant rights to them. Package dependencies retain their own licences. No repository-wide open-source licence is imposed by this packaging step.
