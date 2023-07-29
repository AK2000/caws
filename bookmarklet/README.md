# CAWS Bookmarklet

This folder contains the javascript code used for the bookmarklet. The
bookmarklet query the energy usage of your available endpoints within our database and 
will inject HTML code into the
[!Globus web app's Compute page](https://app.globus.org/compute) to display
energy expenditure information.

## How to use

Create a bookmark of any name with the content of
[!bookmarklet](https://github.com/AK2000/caws/blob/bookmarklet/bookmarklet/bookmarklet)
as the URL. Then, proceed to the [!Globus Compute](https://app.globus.org/compute) page and
click on your newly created bookmark to execute the bookmarklet. The Globus Compute page should then
be updated with the total energy expenditure of the endpoints.

## Server

By default, the code queries our running server for energy expenditure information. However, if you
have chosen to host this information elsewhere, we have provided the server code used by our bookmarklet.
To execute the server, simply define the following environment variables `CAWS_HOST`, `CAWS_USER`,
`CAWS_PASSWORD` and execute the provided [!server.py](https://github.com/AK2000/caws/blob/bookmarklet/bookmarklet/server.py) script.

For example, `CAWS_HOST='127.0.0.1' CAWS_USER='fake_user' CAWS_PASSWORD='fake_password' python server.py`.

