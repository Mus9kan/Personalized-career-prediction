// function toggleSidebar() {
//     const sidebar = document.querySelector('.sidebar');
//     const content = document.querySelector('.content');
  
//     if (sidebar.style.width === '60px') {
//       sidebar.style.width = '250px';
//       content.style.marginLeft = '250px';
//     } else {
//       sidebar.style.width = '60px';
//       content.style.marginLeft = '60px';
//     }
//   }
  // Sidebar toggle functionality
  function regUser() {
    let name = document.querySelector("#name").value;
    let email = document.querySelector("#email").value;

    if (!name || !email || !contactNumber || !password || !cpassword) {
        alert("Please fill in all fields.");
        return;
      }
      let userInfo = {
        name: name,
        email: email,
      };
      const url = "http://localhost:4000/passenger"
      const options = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(userInfo),
      };
      fetch(url, options)
    .then((response) => {
      if (!response.ok) {
        console.error("Registration failed");
        alert("Registration failed. Please try again.");
      } else {
        window.location.href = "crud.html";
      }
    })
    .catch((error) => console.error("Network Error:", error));
}
const loginUser = () => {
    let email = document.querySelector("#name").value;
    let password = document.querySelector("#email").value;
    if (!name || !email) {
        alert("Please fill in all fields.");
        return;
      }
    fetch("http://localhost:4000/passenger")
    .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch user data");
        return response.json();
      })
    };


 function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  sidebar.style.width = sidebar.style.width === '250px' ? '60px' : '250px';
}



// Graph 1: Passengers Over Time
const ctx1 = document.getElementById('passengersOverTime').getContext('2d');
new Chart(ctx1, {
    type: 'bar',
    data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [{
            label: 'Passengers Over Time',
            data: [5, 10, 8, 12, 7, 6, 9],
            backgroundColor: '#27ae60',
            borderRadius: 5,
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 2
                }
            }
        }
    }
});

// Graph 2: Passenger Load Factor
const ctx2 = document.getElementById('passengerLoadFactor').getContext('2d');
new Chart(ctx2, {
    type: 'bar',
    data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [
            {
                label: 'Economy Class',
                data: [4, 8, 6, 10, 5, 7, 8],
                backgroundColor: 'rgb(58, 128, 128)',
                borderRadius: 5,
            },
            {
                label: 'Business Class',
                data: [2, 6, 5, 7, 3, 4, 5],
                backgroundColor: 'pink',
                borderRadius: 10,
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    stepSize: 2
                }
            }
        }
    }
});

