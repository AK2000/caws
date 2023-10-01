# CAWS Bookmarklet

This folder contains the javascript code used for the bookmarklet. The
bookmarklet query the energy usage of your available endpoints within our database and 
will inject HTML code into the
[Globus web app's Compute page](https://app.globus.org/compute) to display
energy expenditure information.

## How to use

Create a bookmark of any name with the content of
[bookmarklet](https://github.com/AK2000/caws/blob/bookmarklet/bookmarklet/bookmarklet)
as the URL. This can also be done via a drag-and-drop of the following [Globus Compute Bookmarklet](javascript:(async()=>{var s=document.querySelectorAll('div[class="h3 my-0"] a[href*="endpoints"]');Array.prototype.forEach.call(s,(async function(s,e){var a=s.href.split("/");let d=`https://129.114.27.161:5001/caws?endpoint=${a.pop()||a.pop()}`;const l=await fetch(d),t=await l.json();var c=s.parentNode.parentNode.nextElementSibling.firstElementChild;if(o=c.querySelector(".caws-energy"),u=c.querySelector(".caws-consumed"),null!=o||null!=u)null!=o&&(o.innerHTML=t.total_energy),null!=u&&(u.innerHTML=t.energy_consumed);else{var n=document.createElement("div");n.classList.add("caws"),n.classList.add("col-lg"),n.classList.add("py-1");var i=document.createElement("div");i.classList.add("row");var r=document.createElement("div");r.classList.add("col-4"),r.classList.add("col-lg-12"),r.classList.add("small"),r.classList.add("text-uppercase"),r.classList.add("text-success"),r.innerHTML='Total node energy expenditure (kWh) <img src="https://www.svgrepo.com/show/923/leaf.svg" alt="leaf" width="13px" heigh="13px" />',i.appendChild(r);var o=document.createElement("div");o.classList.add("caws-energy"),o.classList.add("col-8"),o.classList.add("col-lg-12"),o.classList.add("small"),o.innerHTML=t.total_energy,i.appendChild(o),n.appendChild(i),c.appendChild(n);var p=document.createElement("div");p.classList.add("caws"),p.classList.add("col-lg"),p.classList.add("py-1");var m=document.createElement("div");m.classList.add("row");var L=document.createElement("div");L.classList.add("col-4"),L.classList.add("col-lg-12"),L.classList.add("small"),L.classList.add("text-uppercase"),L.classList.add("text-success"),L.innerHTML='Energy consumed by tasks (kWH) <img src="https://www.svgrepo.com/show/923/leaf.svg" alt="leaf" width="13px" heigh="13px" />',m.appendChild(L);var u=document.createElement("div");u.classList.add("caws-consumed"),u.classList.add("col-8"),u.classList.add("col-lg-12"),u.classList.add("small"),u.innerHTML=t.energy_consumed,m.appendChild(u),p.appendChild(m),c.appendChild(p)}}),s)})();) into your browser's bookmark toolbar. Then, proceed to the [Globus Compute](https://app.globus.org/compute) page and
click on your newly created bookmark to execute the bookmarklet. The Globus Compute page should then
be updated with the total energy expenditure of the endpoints.

## Server

By default, the code queries our running server for energy expenditure information. However, if you
have chosen to host this information elsewhere, we have provided the server code used by our bookmarklet.
To execute the server, simply define the following environment variables `CAWS_HOST`, `CAWS_USER`,
`CAWS_PASSWORD` and execute the provided [server.py](https://github.com/AK2000/caws/blob/bookmarklet/bookmarklet/server.py) script.

For example, `CAWS_HOST='127.0.0.1' CAWS_USER='fake_user' CAWS_PASSWORD='fake_password' python server.py`.

