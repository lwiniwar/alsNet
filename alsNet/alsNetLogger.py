import markdown
import matplotlib
matplotlib.use('agg')
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
    def __init__(self, outfile, training_files=[], num_points=0, multiclass=True, extra=""):
        self.outfile = outfile
        self.startdate = datetime.datetime.now()
        self.arch = None
        self.losses = []
        self.lr = []
        self.points_seen = []
        self.accuracy_train = []
        self.perc_ground = []
        self.perc_building = []
        self.perc_lo_veg = []
        self.perc_med_veg = []
        self.perc_hi_veg = []
        self.perc_water = []
        self.perc_rest = []
        self.cumaccuracy_train = []
        self.valid_points_seen = []
        self.valid_points_acc = []
        self.valid_points_cumacc = []
        self.valid_confusion = []
        self.plots = {}
        self.container = None
        self.training_files = training_files
        self.num_points = num_points
        self.multiclass = multiclass
        self.extra = extra
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

    def add_plot(self):
        pass

    def save(self):
        currdate = datetime.datetime.now()
        train_repr = (self.training_files[:10] + ["..."]) if len(self.training_files) > 10 else self.training_files
        md = """
alsNet Logger
=============
Date started: {startdate}

Current Date: {currdate}

* * *

Parameters
----------

### Global
    
points per batch: {ppb}

learning rate: {learning_rate}

dropout rate: {dropout_rate}

classes: {classes}

training files:

    {training_files}
    
{extra}
        """.format(startdate=self.startdate.strftime('%Y-%m-%d %H:%M:%S'),
                   currdate=currdate.strftime('%Y-%m-%d %H:%M:%S'),
                   learning_rate=self.container.learning_rate,
                   training_files="\n    ".join(train_repr),
                   ppb=self.num_points,
                   dropout_rate=self.container.dropout,
                   classes="all" if self.multiclass else "only ground/non-ground",
                   extra=self.extra)
        for nr, level in enumerate(self.arch):
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

![Instantaneous accuracy]({plot_acc} "Instantaneous accuracy")

Instantaneous accuracy (latest: {acc})

![Cumulative accuracy]({plot_cumacc} "Cumulative accuracy")

Cumulative accuracy (latest: {cumacc})

![Class representativity]({plot_class} "Class representativity")

Class representativity

![Confusion matrix]({plot_confusion} "Confusion matrix")

Confusion matrix

* * *

Testing
-------
N/A
""".format(loss=self.losses[-1],
           acc=self.accuracy_train[-1],
           cumacc=self.cumaccuracy_train[-1],
           plot_acc=self.plots['acc'],
           plot_cumacc=self.plots['cumacc'],
           plot_loss=self.plots['loss'],
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
            'points_seen': self.points_seen,
            'losses': self.losses,
            'lr': self.lr,
            'perc_ground': self.perc_ground,
            'perc_building': self.perc_building,
            'perc_hi_veg': self.perc_hi_veg,
            'perc_lo_veg': self.perc_lo_veg,
            'perc_med_veg': self.perc_med_veg,
            'perc_water': self.perc_water,
            'perc_rest': self.perc_rest,
            'cumaccuracy': self.cumaccuracy_train,
            'accuracy': self.accuracy_train,
            'valid_points': self.valid_points_seen,
            'valid_accuracy': self.valid_points_acc,
            'valid_cumaccuracy': self.valid_points_cumacc
        }
        np.save(os.path.join(data_folder, 'data.npy'), d)

        logging.debug("Starting plotting...")

        fig = plt.figure(figsize=(10,4))
        plt.plot(self.points_seen, self.losses, label='mean loss')
        plt.xlabel("Mio. points seen")
        plt.ylabel("Loss (absolute)")
        ax2 = plt.twinx()
        ax2.plot(self.points_seen, self.lr, color="green", label="learning rate")
        ax2.set_ylabel("Learning rate")
        ax2.set_yscale("log")
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['loss'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                           ).decode('utf-8').replace('\n', '')
        #plt.savefig(os.path.join(outpath, 'plot_loss.png'), bbox_inches='tight')
        plt.close()


        fig = plt.figure(figsize=(10,4))
        plt.plot(self.points_seen, self.accuracy_train, label='current accuracy')
        plt.plot(self.points_seen, self.perc_ground, color='tab:purple', label='ground point percentage')
        plt.plot(self.valid_points_seen, self.valid_points_acc, color='g', marker='+', linestyle='None', label='validation accuracy')
        plt.legend(loc=3)
        plt.xlabel("Mio. points seen")
        plt.ylabel("Percent")
        plt.ylim([0, 100])
        plt.grid(True)
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['acc'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                          ).decode('utf-8').replace('\n', '')
        #plt.savefig(os.path.join(outpath, 'plot_acc.png'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(10,4))
        plt.plot(self.points_seen, self.cumaccuracy_train, label='cumulative accuracy')
        plt.plot(self.valid_points_seen, self.valid_points_cumacc, label='cumulative validation accuracy')
        plt.legend(loc=3)
        plt.xlabel("Mio. points seen")
        plt.ylabel("Percent")
        plt.ylim([0, 100])
        plt.grid(True)
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['cumacc'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                             ).decode('utf-8').replace('\n', '')
        #plt.savefig(os.path.join(outpath, 'plot_cumacc.png'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(10,4))
        plt.stackplot(self.points_seen,
                      self.perc_ground, self.perc_hi_veg, self.perc_med_veg,
                      self.perc_lo_veg, self.perc_building, self.perc_water, self.perc_rest,
                      labels=['ground', 'hi veg', 'med veg', 'lo veg', 'building'],
                      colors=('xkcd:bright purple',
                              'xkcd:dark green',
                              'xkcd:kelly green',
                              'xkcd:lime',
                              'xkcd:light red',
                              'xkcd:water blue',
                              'xkcd:light grey'))

        plt.ylim([0, 100])
        plt.ylabel("Percent")
        #ax2 = plt.twinx()
        #ax2.plot(self.points_seen, self.losses)
        #ax2.set_ylabel("Std. dev. of percent")
        #plt.legend(loc=3)
        plt.xlabel("Mio. points seen")
        plt.tight_layout()
        figdata = BytesIO()
        plt.savefig(figdata, format='png')
        self.plots['classes'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                             ).decode('utf-8').replace('\n', '')
        plt.close()

        # confusion matrix plot
        if self.valid_confusion:
            fig = plt.figure(figsize=(10, 10))
            num_classes = self.valid_confusion[0].shape[0]
            for ref_class in range(num_classes):
                curr_ref_axis = None
                for eval_class in range(num_classes):
                    curplt_id = ref_class * num_classes + eval_class + 1
                    conf_timeline = [self.valid_confusion[i][ref_class, eval_class] for i in range(len(self.valid_confusion))]
                    if curr_ref_axis:
                        plt.subplot(num_classes, num_classes, curplt_id, sharey=curr_ref_axis)
                    else:
                        curr_ref_axis = plt.subplot(num_classes, num_classes, curplt_id)
                    plt.plot(self.valid_points_seen, conf_timeline)
                    plt.ylim([0, 1])

            plt.tight_layout()
            figdata = BytesIO()
            plt.savefig(figdata, format='png')
            self.plots['confusion'] = "data:image/png;base64,%s" % base64.b64encode(figdata.getvalue()
                                                                                  ).decode('utf-8').replace('\n', '')
            plt.close()
        else:
            self.plots['confusion'] = ""

        logging.debug("Plotting done.")