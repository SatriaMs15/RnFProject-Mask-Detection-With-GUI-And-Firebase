import matplotlib
import matplotlib.pyplot as plt
import firebase_config
import pyrebase


def addGraph():
    # array for plot
    date = []
    count = []

    # Initalized firebase
    config = firebase_config.fb_config()
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()

    # Get all the counting data
    data_Date = database.child("Counting").get()

    # Loop for every data and append it to array for plot
    for item in data_Date.each():
        date_get = str(item.key())

        count_get = database.child("Counting").child(date_get).get()
        count_get = int(count_get.val()["count"])

        date.append(date_get)
        count.append(count_get)

    # Make Plot
    matplotlib.rc('font', size=18)
    matplotlib.rc('axes', titlesize=18)

    # Define plot space
    fig, ax = plt.subplots(figsize=(16, 12), dpi=40)

    # Define x and y axes
    ax.bar(date, count, color='darkblue', alpha=0.3)

    # Set plot title and axes labels
    ax.set_ylabel('Count', fontsize=30)

    plt.xticks(rotation=-45)

    return fig
