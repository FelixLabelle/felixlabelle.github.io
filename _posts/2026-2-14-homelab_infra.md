# Homelab for NLP

To carry out experiments in NLP you need compute and to that end I've built out a homelab. This blog 
shouldn't necessarily serve as a blueprint, but rather a framework for considering how to setup a lab
for NLP experiments. I'll set out my needs, considerations, and future steps for the homelab.

## Reasoning for Using a Homelab

In industry if you aren't working for or at one of the big tech companies who 
own their compute using the cloud is par for the course. Through my job I've seen the cost and issues associated with cloud services first hand.
Here are the four primary reasons I decided to build a homelab:

1. Cost control. To give an idea of the cost savings, my training rig cost less than one month of 40GB A100 time and has more VRAM (albeit split across 2 slower GPUs, so this might be a moot point).
2. Privacy and data preservation concerns. You can keep exact copies of datasets and they won't be removed
3. Deeper control over the stack, hardware, software. This matters for experiments as hardware and configuration settings can effect results
4. The homelab heats my office a few degrees in winter 😂

There are some downsides, especially related to uptime, upfront cost, the planning required with scaling, and limits about what can easily be run in a home due 
to physical constraints (e.g., space, power, noise, heat). Given the scale of my experiments and lab I haven't run into these yet. I also don't for see these being 
a major issue, compute seems to be largely under utilized in my experience. I'm not opposed to using the cloud, but more on that later.

## Walking through the homelab

I think when approaching a homelab, the approach should be to aim for minimalism and not 
solve unnecessary problems with cool devices. The homelab may seem like a lot, but it was 
built out over 2-3 years. Every device added solved a specific problem.
At the moment the devices in it are the following:
1. Modem: Connects the network to the outside world; Hitron CODA DOCSIS 3.1
2. Router: Provides wifi TP-Link Dual-Band BE3600. Most machines are hard wired to it
3. db machine: Dell wise 5070; hosts all my DBs and apps that rely on them
4. site machine: Dell wise 5070; hosts different sites
5. NAS: DS224+, NAS, 2 bay DS224+ with 16GB HDDs.
6. Computer UPS: Larger UPS meant to keep non-networking equipment running. Kept separate since these machines 
7. Training machine: Used to train models; [Custom build](https://pcpartpicker.com/list/tWw6Nz)
8. Inference machine: [Framework desktop](https://frame.work/desktop); used to inference models
9. 1GB dummy switch: Netgear switch, connected to the router and most machines
10. Networking UPS (not visible): Kept the lower power machines power separate so they can last longer.


As previously stated, beyond it's composition, what matter more is the homelab's purpose. For me it serves four needs:
1. Storage
2. Training compute
3. Inference compute
4. Application hosting

### Storage

#### Raw Storage

This was the start of the homelab. Having many machines with a amount of large storage 
would be cost prohibitive and distributing data between them would likely need to be a manual effort
or require a specialized tool. My storage needs are:
1. to store datasets
2. backup my computers and application data
3. host a git server

For those reasons I chose a dedicated NAS. I'm happy with a two-bay setup for the NAS. The increased cost of the options with more drives doesn't seem worth while
for now. The one change I would make in the future is swap HDDs for SSDs. HDDs are noisy and data access speeds can be a bottleneck.
That being said SSDs have increased a lot in price recently 
due to the NAND shortage, so I'll keep using HDDs for the foreseeable future.

#### Database

Besides raw data, backups, and code; I also have a fair amount of structure data and databases.
This has translated to having a dedicated DB server.
Since a DB machine will always be on there is an inherent trade off, for me at least, between idle power consumption 
and query speeds. Currently the DBs are running on a souped-up dell wise 5070. It has 32GB of ddr4 soddim ram and 1TB
of nvme 2260. Even with these modifications the dell wyse only has 4 cores, but frankly it's been more than potable for accessing data.
As the number of databases and table size grows, the query speeds especially for text indexing
have slowed to a crawl (I had a table with 22 million spans and queries could take up to a minute). Some of this may be due to sub-optimal 
configuration of the underlying postgresql db (using pg_vector on top). For now, it works wells enough.

### Training

The purpose of this data is ultimately to train models. I'm primarily interested in training encoder-only models. There are two reasons for this 
1. competitive versions of these models can be trained with lower compute requirements compared to decoders (namely gpu vram). There are a number of papers experimenting with this (e.g., [cramming BERT](https://arxiv.org/abs/2212.14034)).
2. IR and improving performance on classification are my primary research interests

Given the upfront cost of a GPU machine scales with GPUs and the fact I'm primarily time bound (research is a nights and weekends thing for me),
I opted to have the a smaller GPU machine run more rather than a beefier machine. For this reason I [built a 2 3090 GPU machine](https://pcpartpicker.com/list/tWw6Nz),
it has enough VRAM to be able to train models, while limiting cost. [For those interested there are 
some pretty hardcore home rigs out there](https://www.ahmadosman.com/blog/serving-ai-from-the-basement-part-i/). Ahmad Osman has some of them and regularly tweets 
advice relating to home builds. I would recommend following his advice rather than mine if you want to build a custom machine.

I'm generally happy with the machine and it's performance, but the GPUs are a bit close together and one of the GPUs heats up about 10C more.
I've devised a fix to funnel air better that minimizes the difference to be ~ 5C and regularly swap the GPUs positions to avoid one 
being exposed to more heat. You could water cool or use a board with better spacing, but frankly it hasn't been enough of an issue 
for me to try and learn how to water cool a system (yet).

Another "odd" choice for the training machine is the use of Windows. I run linux on every other system in the homelab,
and previously this machine used to run Linux as well. However after a driver update one of the GPUs started having PCI bus issues.
At first I thought the card might have died. However after significant debugging (i.e., hair pulling), I realized the drivers had a regression. Swapping to Windows
fixed this ¯\_(ツ)_/¯. It introduces some limitations, but most training frameworks support Windows so it's a viable alternative.

### Inference

While I primarily am interested in encoder models for classification and embedding, I use decoder models for text generation and few-shot classification.
These models tend to be larger in size, in which case the GPU vRAM is the primary chokepoint.
I could've built a larger GPU machine and varied the number of GPUs used for inference, but this is very expensive.
Also given the current mania around models, prices for GPUs have gone up signficantly 
even on the used market. I've seen them for 50% more than 1-2 years ago.

Instead I chose to get an inference 
only machine and reserve the GPU machine for training. Options for building an inference machine are relatively limited at the moment. Inference has two primary bottle necks,
bandwidth for compute and memory. When I did my research (~1 year ago) there were primarily three options:
1. GPU
2. CPU + High-bandwidth memory
3. APU

Below is a table comparing (at a high-level) these 3 options. Because of how many variants there are for each options and the variability 
in computer parts pricing these days, take this table with a grain of salt.

<!-- TODO: Verify the validity of each item -->
<!-- TODO: Add idle power -->
| Dimension               | GPU                                    | CPU + Multichannel DDR5 Memory (server grade)                     | APU                                     |
|-------------------------|----------------------------------------|---------------------------------------|-----------------------------------------|
| **Cost**                | High ($500–$15,000+ per device)        | High ($3,000-$6,000)                  | Moderate ($800–$5000)                         |
| **Peak Energy Consumption**  | High (250W–500W+ per device)           | Moderate (150W–400W)                  | Moderate (150W–200W)                          |
| **Idle Energy Consumption**  | High (30-100W + 25-50W per device)           | Low (30-100W)                  | Low (30-50W)                          |
| **Software Support**    | *Good (*NVIDIA: CUDA, cuDNN, TensorRT)      | Good     | Fair (Limited to vendor frameworks)     |
| **Generation Speed**    | Very Fast (High parallel throughput)   | Slow-Moderate (particularly for large models)| Moderate (Larger models suffer)      |
| **Availability**        | Low-Moderate (Model dependent, supply constrained)          | Moderate (Steady supply, not consumer grade)                 | Moderate (Widely stocked)                  |
| **Memory (GB)**         | 8–80 GB (VRAM)                         | 128–512 GB (RAM)                   | 32-128 GB (Shared RAM/VRAM)              |
| **Memory Bandwidth**    | Very High (300–1,000 GB/s)             | Moderate (150-400 GB/s, *DDR5, multichannel)              | Moderate (200-400 GB/s)              |

Frankly, APUs are a compromise, but they are what made the most sense to me.
Low energy consumption especially at idle, potable speeds, relatively good availability,
acceptable price points. They can't run the largest models at good generation speeds, but frankly only GPUs 
can do this.
Currently in APUs there are 2 options I'm aware of
1. Mac M-series chips
2. Ryzen AI Max+

I chose the Ryzen AI Max 395+. While the Mac m2,3,4 options are good (and in some cases have better performance)
they are significantly more expensive and lock you into the mac ecosystem.
Those and the Ryzen AI's decent performance on sub 30B parameter models were the decision points for me.
Since it's release there has been 
a small, but reliable amount of community activity which has made using the
platform pretty straightforward (assuming you have technical know-how).

Since my purchase the Nvidia DGX has come out, but I'm not sure how that compares in terms of performance.
If you are looking at a homelab, it may be worth considering this option.
For now I've got an option to serve models for the next couple of years. We'll see what changes 
during that time.

### Application hosting

Data, training, and inference are only part of the tooling required for experimentation and research. To evaluate models,
research information, label UIs, chat with models, leverage agents web based tools with UIs are a useful or even necessary. 
Applications that I host include 
1. Label studio, to label data
2. My research management tool, a custom tool that helps find relevant papers and organize them
3. MLFlow, provides experiment tracking
4. WGer, for tracking my workouts (can't all be technical)
5. Pi hole, it serves as my local dns and blocks certain URLs
6. OpenUI, chat interface that provides basic agentic functionalities
7. OpenCode, I've just started experimenting with agents
8. Label studio compatible back end(s) for active learning

The applications run mostly in dockers, but some applications run on the machines directly (i.e., no containerization).
For now a dell wyse has proven effective and low cost, both up front and in terms
of idle/active power usage. Moreover they are cheap and readily available,
which makes having redundant copies easy (more on that later).


## Random Observations

I'm not a gamer, so I don't really need a powerful laptop. However
for running experiments/personal projects I've traditionally opted for laptops with GPUs 
for inferencing and training (encoder only models primarily) due 
to their computational power and portability.

A homelab has made this need obsolete, especially since with remote access 
through a VPN I have access to every device on the go. For that reason I'm currently experimenting
with using a tablet exclusively. That particular tablet is significantly lower cost
than my previous "gaming" laptop. I'll likely keep 
the laptop alive for the foreseeable future or maybe even make 
it into a "blade" for finetuning/inferencing certain models,
but the redundancy of having a more powerful laptop is useful. 
Moreover having a lower cost spare device means that I won't lose access to my homelab and tools if my laptop ever
"shits the bed".

## Potential improvements

I have a list of ideas to improve the lab/fix issues with it. This is more of a dream list assuming infinite time.
Currently I haven't prioritized these issues and I'm not in a particular rush. Urgent needs to tend to bubble 
to the surface.

### Networking

There are two networking issues I'd like to tackle:
1. More robustness for the VPN server, it goes down from time to time
2. Separating my networking further, limiting access between sub nets and limiting congestion.

My plan to address the robustness is setting up a high-availability cluster that will handle all the ins and outs 
of my network (e.g., pi-hole, VPN, etc..). This likely will be done using a combination of 
prox mox and dell wyses, but that's still in the planning phase.

For separating the network I will likely do this through a VLAN, but this is a much 
longer term project. I have limited experience with networking equipment and 
still need to understand my options here.

### Better Backups
<!-- TODO: Review everything below this point -->

With access and flow network improved, backing up the network is the next step.
First I need to finalize backup of all machines, especially the app data. Currently my plan is to have 
a cron job that pushes data a mounted "/app" directory on the NAS.

Another aspect to resilience is having a good backup methodology. Currently I'm just missing a 
remote backup and that is a work in progress.

<!-- Faster storage for raw data, potentially separate fast vs slow access. -->

### Better DB Machine

My current DB machine lags for larger queries. I think this might be 
related to limited RAM speeds (ddr4 2600 iirc), CPU cores (just 4). I think the ideal machine would 
have DDR5 ram, several cores, and a fast access memory.
I'm unsure how to best balance these memory, ram, and CPU power to get 
a budget machine that doesn't have any choke points. Currently
I'm in the process of reanimating an old machine. Part of the process is figuring out 
if I should upgrade this old machine that has DDR4 or buy a used DDR5 machine.

<!--https://docs.paradedb.com/documentation/performance-tuning/overview
/etc/postgresql/X.X/main/postgresql.conf


### Organization

Better physical organization of machines. Currently it's a bit of a mess,
Make setups more reproducible. Currently a downed machine would require manaual replacement.
Have some redundancy but could use more.
-->

### Increased Usage

I'd like to keep my machines running experiments, ideally fully utilizing,
as much as possible. Currently I'm just manually starting jobs.
Tools like slurm may work for this, but I've never configured this.
For now I'm going to keep doing things manually. 

### Monitoring 

Part of fully utilizing the homelab is monitoring it, namely for stats
like memory, or device heat (potentially to identify problems) relating to 
excessive usage. Grafana & Prometheus seem like good candidates 
to handle this task. When usage dips below a certain amount I could 
trigger emails to my account using a locally hosted SMTP server,
to remind me to start jobs (until I get a queuing tool implemented.

### Robustness

Having usage is great, but what if I have an outage.
I'm working on building a better inventory of spare parts and machines to reduce down time.
The low cost of dell wyse 5070s has made this a possibility. For my custom machines this has translated 
to consolidating parts and making sure there is inter machine compatibility so I can quickly swap parts 
to test and get things running again.

Another component to down time, is the time to setup machines.
I manually replace machines and flash them using a USB at the moment.
I want to have a better method for managing software on machines, maybe something
like ansible?

For software, HA clusters might be worth while. For now, the only software I've 
had issues with is my VPN server, so I may skip this for now.

### Better Method for Tracking System Notes and Information

I currently use [Zettlekasten](https://github.com/Zettelkasten-Team/Zettelkasten/tree/main)
to track what I learn and notes about the homelab. I haven't been the most religious
about documenting everything, but it contains a lot of important information such 
as setup instructions, configs, etc..

I'd like to get a more universal, and ultimately, portable system for keeping track of my notes.
I've heard good things about Obsidian, but migrating hasn't been an urgent issue.

### Beyond the Homelab

Use of the cloud and a homelab don't need to be mutually exclusive, for example data can be preprocessed locally in a way that makes recovering the 
information easier and then trained or run on the cloud. Hashing is a good example of destructive operation that could be used to 
do this while preserving information that would be relevant for a downstream task (e.g., lsh-based deduping).

When the homelab no longer cuts it, I'll need to start planning how to undertake a hybridized approach. For now the idea is that the cloud's 
primary benefit is to offload low cost, low privacy processes that need to be robust to the cloud.
Preprocessing certain data, especially private data locally with light representations 
that make the data hard to recreate or decipher without losing critical information.
The cloud could be used to train or inference on featurized data (featurize locally,
compute rest on the cloud).

##  Closing Remarks

<!-- THere are more multiple narrative threads in here, lets fix that -->
There are a couple of points to improve the homelab with, but ultimately I'm 
trying to avoid maintenance and upgrades becoming a Sisyphean task. The homelab is a means to an end rather than the 
place I spend all my time. For now it's more than adequate for my purposes. Hopefully if doing computationally intensive experiments at home, or elsewhere,
is something you are interested in this guide gave you food for thought. Feel free to comment if you have any questions.