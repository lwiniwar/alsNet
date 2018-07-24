import markdown
import matplotlib
matplotlib.use('agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import datetime
import codecs
import os
import logging

HEAD = """
<html>
<head>
<style>
body{
    margin: 0 auto;
    font-family: Georgia, Palatino, serif;
    color: #444444;
    line-height: 1;
    max-width: 960px;
    padding: 30px;
}
h1, h2, h3, h4 {
    color: #111111;
    font-weight: 400;
}
h1, h2, h3, h4, h5, p {
    margin-bottom: 24px;
    padding: 0;
}
h1 {
    font-size: 48px;
}
h2 {
    font-size: 36px;
    margin: 24px 0 6px;
}
h3 {
    font-size: 24px;
}
h4 {
    font-size: 21px;
}
h5 {
    font-size: 18px;
}
a {
    color: #0099ff;
    margin: 0;
    padding: 0;
    vertical-align: baseline;
}
ul, ol {
    padding: 0;
    margin: 0;
}
li {
    line-height: 24px;
}
li ul, li ul {
    margin-left: 24px;
}
p, ul, ol {
    font-size: 16px;
    line-height: 24px;
    max-width: 540px;
}
pre {
    padding: 0px 24px;
    max-width: 800px;
    white-space: pre-wrap;
}
code {
    font-family: Consolas, Monaco, Andale Mono, monospace;
    line-height: 1.5;
    font-size: 13px;
}
aside {
    display: block;
    float: right;
    width: 390px;
}
blockquote {
    margin: 1em 2em;
    max-width: 476px;
}
blockquote p {
    color: #666;
    max-width: 460px;
}
hr {
    width: 540px;
    text-align: left;
    margin: 0 auto 0 0;
    color: #999;
}
table {
    border-collapse: collapse;
    margin: 1em 1em;
    border: 1px solid #CCC;
}
table thead {
    background-color: #EEE;
}
table thead td {
    color: #666;
}
table td {
    padding: 0.5em 1em;
    border: 1px solid #CCC;
}
</style>
"""
TITLE="""
<title>
{name}
</title>
</head>
<body>
"""

FOOT="""
</body>
</html>
"""

class Logger():
    def __init__(self, outfile, inst, training_files):
        self.outfile = outfile
        self.inst = inst
        self.startdate = datetime.datetime.now()
        self.training_files = training_files
        self.extra = ""
        self.plots = {}

        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))


    def save(self):
        currdate = datetime.datetime.now()
        train_repr = ([f.file for f in self.training_files[:10]]+ ["..."]) if len(self.training_files) > 10 else [f.file for f in self.training_files]
        md = """
alsNet Logger2
==============
Date started: {startdate}

Current Date: {currdate}

* * *

Parameters
----------

### Global
    
points per batch: {ppb}

learning rate: {learning_rate}

dropout rate: {dropout_rate}


training files:

    {training_files}
    
{extra}
        """.format(startdate=self.startdate.strftime('%Y-%m-%d %H:%M:%S'),
                   currdate=currdate.strftime('%Y-%m-%d %H:%M:%S'),
                   learning_rate=self.inst.learning_rate,
                   training_files="\n    ".join(train_repr),
                   ppb=self.inst.num_points,
                   dropout_rate=self.inst.dropout,
                   extra=self.extra)
        for nr, level in enumerate(self.inst.arch):
            md += """
### Level {levelno}

    nPoints = {npoint}
    radius = {radius}
    nSample = {nsample}
    mlp = {mlp}
    pooling = {pooling}
    mlp2 = {mlp2}
    reverse_mlp = {reverse_mlp}
      
            """.format(levelno=nr, **level)


        self.create_plots()
        md += """
* * *

Training
--------

![Loss]({plot_loss} "Loss")

Loss (latest: {loss})

![Accuracy]({plot_acc} "Accuracy")

Accuracy (latest 5: {acc})

![Class representativity]({plot_class} "Class representativity")

Class representativity

![Confusion matrix]({plot_confusion} "Confusion matrix")

Confusion matrix

* * *

Testing
-------
N/A
""".format(loss=self.inst.train_history.losses[-1],
           acc=self.inst.train_history.get_oa_timeline()[-1],

           plot_loss=self.plots['loss'],
           plot_acc=self.plots['acc'],
           plot_class=self.plots['classes'],
           plot_confusion=self.plots['confusion'])

        html = markdown.markdown(md)
        output_file = codecs.open(self.outfile, "w",
                                  encoding="utf-8",
                                  errors="xmlcharrefreplace")
        output_file.write(HEAD + TITLE.format(name=os.path.dirname(self.outfile).split(os.sep)[-1]) + html + FOOT)


    def create_plots(self):
        data_folder = os.path.join(os.path.dirname(self.outfile), "plot_data")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        d = {
            'train_cm': self.inst.train_history.cm,
            'train_losses': self.inst.train_history.losses,
            'train_point_seen': self.inst.train_history.points_seen,
            'train_timestamps': self.inst.train_history.timestamps,
            'eval_cm':  self.inst.eval_history.cm,
            'eval_point_seen': self.inst.eval_history.points_seen,
            'eval_timestamps': self.inst.eval_history.timestamps,
        }
        np.save(os.path.join(data_folder, 'data.npy'), d)

        logging.debug("Starting plotting...")

        fig = plt.figure(figsize=(10,4))
        plt.plot(self.inst.train_history.points_seen, self.inst.train_history.losses, label='mean loss')
        plt.xlabel("Mio. points seen")
        plt.ylabel("Loss (absolute)")
        #ax2 = plt.twinx()
        #ax2.plot(self.points_seen, self.lr, color="green", label="learning rate")
        #ax2.set_ylabel("Learning rate")
        #ax2.set_yscale("log")
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['loss'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                           ).decode('utf-8').replace('\n', '')
        plt.close()


        fig = plt.figure(figsize=(10,4))
        plt.plot(self.inst.train_history.points_seen,
                 self.inst.train_history.get_oa_timeline(),
                 label='current accuracy')
        #plt.plot(self.inst.train_history.points_seen,
        #         self.inst.train_history.get_oa_timeline_smooth(5)[1:-1],
        #         color='tab:purple', label='averaged accuracy (5)')
        plt.plot(self.inst.eval_history.points_seen,
                 self.inst.eval_history.get_oa_timeline(),
                 color='g', marker='+', linestyle='None', label='validation accuracy')

        plt.legend(loc=3)
        plt.xlabel("Mio. points seen")
        plt.ylabel("x100 Percent")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['acc'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                          ).decode('utf-8').replace('\n', '')
        plt.close()

        fig = plt.figure(figsize=(10,4))

        keep_classes = (2, 3, 4, 5, 6, 9)
        for class_id, label, color in zip(keep_classes,
                                          ['ground', 'hi veg', 'med veg', 'lo veg', 'building', 'water'],
                                          ['xkcd:bright purple',
                                           'xkcd:dark green',
                                           'xkcd:kelly green',
                                           'xkcd:lime',
                                           'xkcd:light red',
                                           'xkcd:water blue']
                                          ):
            plt.plot(self.inst.train_history.points_seen,
                     np.cumsum(self.inst.train_history.class_points_timeline(class_id)),
                     label=label,
                     color=color)

        plt.ylabel("Points of class seen")
        plt.legend(loc=3)
        plt.xlabel("Mio. points seen")
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['classes'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                              ).decode('utf-8').replace('\n', '')
        plt.close()

        # confusion matrix plot
        fig = plt.figure(figsize=(10, 10))
        num_classes = len(keep_classes)+1
        keep_classes_e = keep_classes + (-1,)
        gs = gridspec.GridSpec(num_classes, num_classes)

        row = -1
        for ref_class in keep_classes_e:
            curr_ref_axis = None
            row += 1
            col = -1
            for eval_class in keep_classes_e:
                col += 1



                conf_timeline = self.inst.eval_history.get_cm_timeline_compressed(ref_class, eval_class, keep_classes)
                if curr_ref_axis:
                    plt.subplot(gs[row, col], sharey=curr_ref_axis)
                else:
                    curr_ref_axis = plt.subplot(gs[row, col])

                plt.plot(self.inst.eval_history.points_seen, conf_timeline)

                if col == row:
                    plt.gca().set_facecolor('xkcd:pale green')
                    highcolor = 'xkcd:forest green'
                    lowcolor = 'xkcd:grass green'
                else:
                    plt.gca().set_facecolor('xkcd:pale pink')
                    highcolor = 'xkcd:orange red'
                    lowcolor = 'xkcd:dirty pink'
                if conf_timeline:
                    plt.text((self.inst.eval_history.points_seen[0] + self.inst.eval_history.points_seen[-1])/2,
                             0.5,
                             "%.1f%%" % (conf_timeline[-1]*100), ha='center',
                             color=highcolor if conf_timeline[-1]>0.5 else lowcolor)
                    cm = self.inst.eval_history.cm[-1]
                    ref_sum = np.sum(cm, axis=1)[ref_class]
                    eval_sum = np.sum(cm, axis=0)[eval_class]
                    plt.text((self.inst.eval_history.points_seen[0] + self.inst.eval_history.points_seen[-1])/2,
                             0.3,
                             "%d" % (cm[ref_class, eval_class]), ha='center')
                    if col == 0:
                        plt.ylabel('%d\n%d\n(%.0f%%)' % (ref_class,
                                                         ref_sum,
                                                         ref_sum/self.inst.num_points * 100))
                    if row == 0:
                        plt.gca().xaxis.set_label_position('top')
                        plt.xlabel('%d\n%d\n(%.0f%%)' % (eval_class,
                                                         eval_sum,
                                                         eval_sum/self.inst.num_points * 100))


                plt.gca().get_yaxis().set_ticks([])
                plt.gca().get_xaxis().set_ticks([])

                plt.ylim([0, 1])

        fig.text(0.5, 0.94, 'Estimated', ha='center', va='center')
        fig.text(0.06, 0.5, 'Ground truth', ha='center', va='center', rotation='vertical')

        plt.subplots_adjust(hspace=.0, wspace=.0)
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['confusion'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                                ).decode('utf-8').replace('\n', '')
        plt.close()

        logging.debug("Plotting done.")