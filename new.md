## Rewards

This is an example of reward structure used for a two stage RL for getting up motion. Complete RL though, not like our optimization based retargeting. Also, this is for Unitree G1 robot.

### A-1 Rewards components in Stage I

Detailed reward components used in Stage I are summarized in Table II.

TABLE II: **Reward components and weights in Stage I.** Penalty rewards prevent undesired behaviors for sim-to-real transfer, regularization refines motion, and task rewards ensure successful getting up or rolling over.

|   |   |   |
|---|---|---|
|**TERM**|**EXPRESSION**|**WEIGHT**|
|**Penalty:**|||
|Torque limits|$\mathbb{1}(\boldsymbol{\tau}_t \notin [\boldsymbol{\tau}_{\text{min}}, \boldsymbol{\tau}_{\text{max}}])$|-0.1|
|DoF position limits|$\mathbb{1}(\boldsymbol{d}_t \notin [\boldsymbol{q}_{\text{min}}, \boldsymbol{q}_{\text{max}}])$|-5|
|Energy|$\Vert{} \boldsymbol{\tau} \odot \dot{\boldsymbol{q}} \Vert{}$|-1e-4|
|Termination|$\mathbb{1}_{\text{termination}}$|-500|
|**Regularization:**|||
|DoF acceleration|$\Vert{} \ddot{\boldsymbol{d}}_t \Vert{}_2$|-1e-7|
|DoF velocity|$\Vert{} \dot{\boldsymbol{d}}_t \Vert{}_2^2$|-1e-4|
|Action rate|$\Vert{} \boldsymbol{a}_t \Vert{}_2^2$|-0.1|
|Torque|$\Vert{} \boldsymbol{\tau}_t \Vert{}$|-6e-7|
|DoF position error|$\mathbb{1}(\boldsymbol{h}_{\text{base}} \geq 0.8) \cdot \text{exp}(-0.05 \Vert{} \boldsymbol{d}_t - \boldsymbol{d}_t^{\text{default}} \vert{})$|-0.75|
|Angular velocity|$\Vert{} \boldsymbol{\omega}^2 \Vert{}$|-0.1|
|Base velocity|$\Vert{} \boldsymbol{v}^2 \Vert{}$|-0.1|
|Foot slip|$\mathbb{1}(\boldsymbol{F}_z^{\text{feet}} > 5.0) \cdot \sqrt{\Vert{} \boldsymbol{v}_z^{\text{feet}} \Vert{}}$|-1|
|**Getting-Up Task Rewards:**|||
|Base height exp|$\text{exp}(\boldsymbol{h}^{\text{base}}) - 1$|5|
|Head height exp|$\text{exp}(\boldsymbol{h}^{\text{head}}) - 1$|5|
|$\Delta$ base height|$\mathbb{1}(\boldsymbol{h}_t^{\text{base}} > \boldsymbol{h}_{t-1}^{\text{base}})$|1|
|Feet contact forces reward|$\mathbb{1}(\Vert{} \boldsymbol{F}_t^{\text{feet}} \Vert{} > \Vert{} \boldsymbol{F}_{t-1}^{\text{feet}} \Vert{})$|1|
|Standing on feet reward|$\mathbb{1}((\Vert{} \boldsymbol{F}^{\text{feet}} \Vert{} > 0) \ \& \ (\boldsymbol{h}^{\text{feet}} < 0.2))$|2.5|
|Body upright reward|$\text{exp}(-\mathbf{g}_z^{\text{base}})$|0.25|
|Feet height reward|$\text{exp}(-10 \cdot \boldsymbol{h}^{\text{feet}})$|2.5|
|Feet distance reward|$\frac{1}{2}\big(\text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{min}}, -0.5)\vert{})$<br><br>  <br><br>$+ \text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{max}}, 0)\vert{})\big)$|2|
|Foot orientation|$\sqrt{\Vert{} \mathbf{G}_{xy}^{\text{feet}} \Vert{}}$|-0.5|
|Soft body symmetry penalty|$\Vert{} \mathbf{a}_{\text{left}} - \mathbf{a}_{\text{right}} \Vert{}$|-1.0|
|Soft waist symmetry penalty|$\Vert{} \mathbf{a}^{\text{waist}} \Vert{}$|-1.0|
|**Rolling-Over Task Rewards:**|||
|Base Gravity Exp|$\frac{1}{2} \big( \text{exp}(-0.01 (1 - \cos \theta_{\text{left}})) +$<br><br>  <br><br>$\text{exp}(-0.01 (1 - \cos \theta_{\text{right}})) \big)$,<br><br>  <br><br>$\cos \theta = \frac{\mathbf{g}^{\text{knee}} \cdot \mathbf{g}_{\text{target}}}{\Vert{}\mathbf{g}^{\text{base}}\Vert{} \Vert{}\mathbf{g}_{\text{base}}\Vert{}}$|8|
|Knee Gravity Exp|$\frac{1}{2} \big( \text{exp}(-0.01 (1 - \cos \theta_{\text{left}})) +$<br><br>  <br><br>$\text{exp}(-0.01 (1 - \cos \theta_{\text{right}})) \big)$,<br><br>  <br><br>$\cos \theta = \frac{\mathbf{g}^{\text{knee}} \cdot \mathbf{g}_{\text{target}}}{\Vert{}\mathbf{g}^{\text{base}}\Vert{} \Vert{}\mathbf{g}_{\text{base}}\Vert{}}$|8|
|Feet distance reward|$\frac{1}{2}\big(\text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{min}}, -0.5)\vert{})$<br><br>  <br><br>$+ \text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{max}}, 0)\vert{})\big)$|2|
|Feet height reward|$\text{exp}(-10 \cdot \boldsymbol{h}^{\text{feet}})$|2.5|

### A-2 Rewards components in Stage II

Detailed reward components used in Stage II are summarized in Table III.

TABLE III: **Reward components and weights in Stage II.** Penalty rewards prevent undesired behaviors for sim-to-real transfer, regularization refines motion, and task rewards ensure successful whole-body tracking in real time.

|   |   |   |
|---|---|---|
|**TERM**|**EXPRESSION**|**WEIGHT**|
|**Penalty:**|||
|Torque limits|$\mathbb{1}(\boldsymbol{\tau}_t \notin [\boldsymbol{\tau}_{\text{min}}, \boldsymbol{\tau}_{\text{max}}])$|-5|
|Ankle torque limits|$\mathbb{1}(\boldsymbol{\tau}_t^{\text{ankle}} \notin [\boldsymbol{\tau}_{\text{min}}^{\text{ankle}}, \boldsymbol{\tau}_{\text{max}}^{\text{ankle}}])$|-0.01|
|Upper torque limits|$\mathbb{1}(\boldsymbol{\tau}_t^{\text{upper}} \notin [\boldsymbol{\tau}_{\text{min}}^{\text{upper}}, \boldsymbol{\tau}_{\text{max}}^{\text{upper}}])$|-0.01|
|DoF position limits|$\mathbb{1}(\boldsymbol{d}_t \notin [\boldsymbol{q}_{\text{min}}, \boldsymbol{q}_{\text{max}}])$|-5|
|Ankle DoF position limits|$\mathbb{1}(\boldsymbol{d}_t^{\text{ankle}} \notin [\boldsymbol{q}_{\text{min}}^{\text{ankle}}, \boldsymbol{q}_{\text{max}}^{\text{ankle}}])$|-5|
|Upper DoF position limits|$\mathbb{1}(\boldsymbol{d}_t^{\text{upper}} \notin [\boldsymbol{q}_{\text{min}}^{\text{upper}}, \boldsymbol{q}_{\text{max}}^{\text{upper}}])$|-5|
|Energy|$\Vert{} \boldsymbol{\tau} \odot \dot{\boldsymbol{q}} \Vert{}$|-1e-4|
|Termination|$\mathbb{1}_{\text{termination}}$|-50|
|**Regularization:**|||
|DoF acceleration|$\Vert{} \ddot{\boldsymbol{d}}_t \Vert{}_2$|-1e-7|
|DoF velocity|$\Vert{} \dot{\boldsymbol{d}}_t \Vert{}_2^2$|-1e-3|
|Action rate|$\Vert{} \boldsymbol{a}_t \Vert{}_2^2$|-0.1|
|Torque|$\Vert{} \boldsymbol{\tau}_t \Vert{}$|-0.003|
|Ankle torque|$\Vert{} \boldsymbol{\tau}_t^{\text{ankle}} \Vert{}$|-6e-7|
|Upper torque|$\Vert{} \boldsymbol{\tau}_t^{\text{upper}} \Vert{}$|-6e-7|
|Angular velocity|$\Vert{} \boldsymbol{\omega}^2 \Vert{}$|-0.1|
|Base velocity|$\Vert{} \boldsymbol{v}^2 \Vert{}$|-0.1|
|**Tracking Rewards:**|||
|Tracking DoF position|$\text{exp}\left(-\frac{(\boldsymbol{d}_t - \boldsymbol{d}_t^{\text{target}})^2}{4}\right)$|8|
|Feet distance reward|$\frac{1}{2}\big(\text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{min}}, -0.5)\vert{})$<br><br>  <br><br>$+ \text{exp}(-100 \ \vert{}\text{max}(\boldsymbol{d}_{\text{feet}} - \boldsymbol{d}_{\text{max}}, 0)\vert{})\big)$|2|
|Foot orientation|$\sqrt{\Vert{} \mathbf{G}_{xy}^{\text{feet}} \Vert{}}$|-0.5|